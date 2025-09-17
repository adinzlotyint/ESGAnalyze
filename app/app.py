import gradio as gr
import json
import re
import fitz
import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import pipeline, AutoTokenizer
import pathlib

# --- 1. KONFIGURACJA ---
MODEL_NAME = "Adinzlotyint/ESGAnalyze"
MAX_LENGTH = 4096
STRIDE = 512

ALL_CRITERIA_CONFIG = [
    {'key': 'c1_transition_plan', 'display': 'C1 - Plan transformacji', 'type': 'AI'},
    {'key': 'c2_risk_management', 'display': 'C2 - Zarządzanie ryzykami', 'type': 'AI'},
    {'key': 'Criteria3', 'display': 'C3 - Poziom dojrzałości ujawnień GHG', 'type': 'Regex'},
    {'key': 'c4_boundaries', 'display': 'C4 - Granice konsolidacji', 'type': 'AI'},
    {'key': 'Criteria5', 'display': 'C5 - Zastosowanie standardu obliczeniowego', 'type': 'Regex'},
    {'key': 'c6_historical_data', 'display': 'C6 - Dane historyczne', 'type': 'AI'},
    {'key': 'c7_intensity_metrics', 'display': 'C7 - Wskaźniki intensywności', 'type': 'AI'},
    {'key': 'c8_targets_credibility', 'display': 'C8 - Wiarygodność celów', 'type': 'AI'},
    {'key': 'Criteria9', 'display': 'C9 - Zastosowanie jednostki miary CO2e', 'type': 'Regex'}
]

# --- 2. ŁADOWANIE ZASOBÓW ---
print("Sprawdzanie dostępności GPU...")
if torch.cuda.is_available():
    print(f"✅ Znaleziono GPU: {torch.cuda.get_device_name(0)}")
    DEVICE_ID = 0
else:
    print("⚠️ Nie znaleziono GPU z obsługą CUDA. Uruchamianie na CPU.")
    DEVICE_ID = -1

print("Ładowanie modelu AI i tokenizera...")
esg_pipeline = pipeline("text-classification", model=MODEL_NAME, device=0, top_k=None, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Ładowanie plików konfiguracyjnych...")

script_directory = pathlib.Path(__file__).parent.resolve()
regex_file_path = script_directory / 'regex.json'

with open(regex_file_path, 'r', encoding='utf-8-sig') as f:
    REGEX_RULES = json.load(f)

thresholds_path = hf_hub_download(repo_id=MODEL_NAME, filename="optimal_thresholds.json")
with open(thresholds_path, 'r', encoding='utf-8') as f:
    OPTIMAL_THRESHOLDS = json.load(f)

AI_CRITERIA_KEYS = [c['key'] for c in ALL_CRITERIA_CONFIG if c['type'] == 'AI']
CRITERIA_TO_IDX = {name: i for i, name in enumerate(AI_CRITERIA_KEYS)}
LABEL_TO_INTERNAL_NAME = {f"LABEL_{i}": name for i, name in enumerate(AI_CRITERIA_KEYS)}

print("✅ Wszystkie zasoby załadowane pomyślnie.")

# --- 3. KLUCZOWE FUNKCJE LOGICZNE ---

def extract_text_from_pdf(pdf_file_obj):
    try:
        doc = fitz.open(pdf_file_obj.name, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        raise gr.Error(f"Błąd podczas przetwarzania pliku PDF: {e}")

def run_regex_module(text, rules, regex_keys):
    results = {}
    for key in regex_keys:
        criterion_rules = rules.get(key, {})
        best_score = 0.0
        for score_str in sorted(criterion_rules.keys(), key=float, reverse=True):
            sub_rules = criterion_rules[score_str]
            for sub_rule_key in sub_rules:
                rule_set = sub_rules[sub_rule_key]
                include = rule_set.get("Include", [])
                exclude = rule_set.get("Exclude", [])

                if all(re.search(p, text, re.IGNORECASE) for p in include) and \
                   not any(re.search(p, text, re.IGNORECASE) for p in exclude):
                    best_score = float(score_str)
                    break
            if best_score > 0:
                break
        results[key] = best_score
    return results

def run_ai_module(text, progress):
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    
    text_chunks = []
    for i in range(0, len(tokens), STRIDE):
        chunk_tokens = tokens[i : i + MAX_LENGTH]
        if len(chunk_tokens) > 10:
            text_chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

    if not text_chunks:
        return {}
    
    batch_size = 16
    all_predictions = []
    num_chunks = len(text_chunks)
    
    progress(0, desc="Analiza AI dokumentu...")
    for i in range(0, num_chunks, batch_size):
        batch = text_chunks[i : i + batch_size]
        predictions = esg_pipeline(batch, truncation=True, max_length=MAX_LENGTH)
        all_predictions.extend(predictions)
        progress((i + len(batch)) / num_chunks, desc=f"Przetwarzanie fragmentów {i+len(batch)}/{num_chunks}")

    chunk_scores = np.zeros((num_chunks, len(AI_CRITERIA_KEYS)))
    for i, preds in enumerate(all_predictions):
        for pred in preds:
            internal_label = LABEL_TO_INTERNAL_NAME.get(pred['label'])
            if internal_label and internal_label in CRITERIA_TO_IDX:
                label_idx = CRITERIA_TO_IDX[internal_label]
                chunk_scores[i, label_idx] = pred['score']
    
    aggregated_scores = np.percentile(chunk_scores, 75, axis=0)
    
    final_results = {}
    for i, key in enumerate(AI_CRITERIA_KEYS):
        final_results[key] = aggregated_scores[i]
        
    return final_results

def calculate_confidence_score(score, threshold):
    if score >= threshold:
        return 0.5 + 0.5 * (score - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
    else:
        return 0.5 * (score / threshold) if threshold > 0.0 else 0.5

# --- 4. GŁÓWNA FUNKCJA ORKIESTRUJĄCA ---

def analyze_report(pdf_file, progress=gr.Progress(track_tqdm=True)):
    if pdf_file is None:
        return pd.DataFrame()

    progress(0, desc="Wczytywanie i ekstrakcja tekstu z PDF...")
    text = extract_text_from_pdf(pdf_file)
    
    ai_scores = run_ai_module(text, progress)
    
    progress(1.0, desc="Analiza regułowa i formatowanie wyników...")
    regex_keys = [c['key'] for c in ALL_CRITERIA_CONFIG if c['type'] == 'Regex']
    regex_scores = run_regex_module(text, REGEX_RULES, regex_keys)
    
    output_data = []
    for criterion in ALL_CRITERIA_CONFIG:
        key, display_name, type = criterion['key'], criterion['display'], criterion['type']
        
        ocena_str = ""
        result_str = ""
        module_str = ""

        if type == 'AI':
            score = ai_scores.get(key, 0.0)
            threshold = OPTIMAL_THRESHOLDS.get(key, 0.5)
            decision = 1 if score >= threshold else 0
            confidence = calculate_confidence_score(score, threshold)
            
            ocena_str = "✅ Spełnione" if decision == 1 else "❌ Niespełnione"
            result_str = f"{confidence:.1%} (Wynik: {score:.2%})"
            module_str = "AI"
        else:
            score = regex_scores.get(key, 0.0)
            result_str = f"{score:.2f}"
            module_str = "Regułowy"
            
            if score == 1.0:
                ocena_str = "✅ Spełnione"
            elif score > 0.0:
                ocena_str = "⚠️ Częściowo spełnione"
            else:
                ocena_str = "❌ Niespełnione"

        output_data.append({
            "Kryterium": display_name, 
            "Ocena": ocena_str,
            "Pewność / Wynik": result_str,
            "Moduł": module_str
        })
        
    return pd.DataFrame(output_data)

# --- 5. DEFINICJA INTERFEJSU GRADIO ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Hybrydowy System Weryfikacji Raportów ESG")
    gr.Markdown("Prosta i skuteczna analiza raportów z wykorzystaniem AI i precyzyjnych reguł.")
    
    pdf_input = gr.File(label="Wgraj raport ESG (PDF)", file_types=[".pdf"])
    analyze_button = gr.Button("Analizuj Raport", variant="primary")
    
    gr.Markdown(
        "**💡 Jak interpretować wyniki i czego się spodziewać?**\n"
        " - **AI:** Ocena 'Spełnione' przyznawana jest na podstawie progu pewności. W nawiasie surowy wynik z modelu.\n"
        " - **Regułowy:** Wyświetlany jest wynik (0.00-1.00) oraz ocena: Spełnione (dla 1.0), Częściowo spełnione (dla >0) lub Niespełnione (dla 0).\n"
        " - **Wydajność:** Analiza bardzo dużych raportów (>100 stron) **może zająć nawet kilkanaście minut**. Wynika to ze skomplikowanej operacji dzielenia dokumentu na setki fragmentów dla modelu AI i jest normalnym zachowaniem dla tej wersji demonstracyjnej.\n"
        " - **Jakość danych (Zasada GIGO):** System jest przeznaczony dla autentycznych raportów ESG. Wyniki dla plików o niskiej jakości (np. skanów, plików z losowym tekstem) mogą być nieprzewidywalne, co jest zgodne z zasadą **'garbage in, garbage out' (śmieci na wejściu, śmieci na wyjściu)**."
    )
    
    output_df = gr.DataFrame(headers=["Kryterium", "Ocena", "Pewność / Wynik", "Moduł"], label="Szczegółowa ocena kryteriów")

    gr.Markdown(
        "**Opis kryteriów:**\n"
        " - **C1 - Plan transformacji:** Kryterium ocenia, czy raport w sposób jednoznaczny przedstawia plan transformacji na potrzeby łagodzenia zmiany klimatu, który jest zintegrowany z ogólną strategią biznesową i modelem biznesowym spółki. Weryfikowane jest, czy firma nie tylko deklaruje posiadanie polityki klimatycznej, ale również opisuje, w jaki sposób jej cele i działania klimatyczne przekładają się na kluczowe aspekty operacyjne i strategiczne.\n"
        " - **C2 - Zarządzanie ryzykami:** Kryterium ocenia, czy raport, poza identyfikacją istotnych ryzyk i szans klimatycznych, przedstawia również wdrożone polityki, procesy lub działania służące do zarządzania nimi. Nacisk kładziony jest na wykazanie aktywnego podejścia do mitygacji ryzyk i adaptacji, w odróżnieniu od jedynie pasywnej świadomości ich istnienia.\n"
        " - **C3 - Poziom dojrzałości ujawnień GHG:** \n"
                    "*Poziom Podstawowy (0.25 pkt)*: Identyfikacja obecności danych dla Zakresu 1 i 2. \n"
                    "*Poziom Zgodności Metodologicznej (0.5 pkt)*: Weryfikacja, czy dla Zakresu 2 raportowane są dane z wykorzystaniem obu wymaganych metod (rynkowej i lokalizacyjnej). \n"
                    "*Poziom Kompletności Zakresów (0.75 pkt)*: Sprawdzenie, czy raport, oprócz Zakresu 1 i 2, zawiera również dane dotyczące Zakresu 3, obejmującego łańcuch wartości. \n"
                    "*Poziom Zaawansowany (1 pkt)*: Potwierdzenie, że raportowanie Zakresu 3 jest dodatkowo uszczegółowione poprzez wskazanie konkretnych kategorii emisji (zgodnie z 15 kategoriami GHG Protocol).\n"
        " - **C4 - Granice konsolidacji:** Kryterium weryfikuje, czy organizacja w sposób transparentny definiuje granice organizacyjne dla raportowanych emisji GHG. Pozytywna ocena wymaga, aby raport jasno określał, które podmioty (np. cała grupa kapitałowa, wybrane spółki zależne) zostały włączone do kalkulacji, a w przypadku jakichkolwiek wyłączeń – przedstawiał jasne uzasadnienie i kryteria takiego wyboru. Brak zdefiniowanych granic lub arbitralne, nieuzasadnione wyłączenia są oceniane negatywnie.\n"
        " - **C5 - Zastosowanie standardu obliczeniowego:** Kryterium weryfikuje, czy raport zapewnia przejrzystość metodologiczną w zakresie obliczania emisji GHG poprzez wskazanie źródeł, z których pochodzą zastosowane wskaźniki emisji oraz współczynniki potencjału tworzenia efektu cieplarnianego (GWP). Ocena jest dokonywana za pomocą systemu opartego na regułach, który identyfikuje w tekście odwołania do uznanych, międzynarodowych lub krajowych baz danych i standardów (takich jak IPCC, GHG Protocol, DEFRA, KOBiZE). Pozytywna ocena jest przyznawana raportom, które jasno deklarują podstawę swoich kalkulacji, co jest fundamentalnym warunkiem weryfikowalności i porównywalności prezentowanych danych emisyjnych.\n"
        " - **C6 - Dane historyczne:** Kryterium ocenia, czy raport dostarcza wystarczających danych historycznych do przeprowadzenia wiarygodnej analizy trendów emisyjnych i oceny postępów w realizacji celów klimatycznych. Pozytywna ocena jest przyznawana raportom, które prezentują spójny szereg czasowy danych o emisjach GHG obejmujący co najmniej trzy ostatnie lata (rok sprawozdawczy i dwa lata poprzedzające). Prezentacja danych jedynie za bieżący lub dwa ostatnie lata jest uznawana za niewystarczającą do rzetelnej oceny długoterminowej dynamiki zmian.\n"
        " - **C7 - Wskaźniki intensywności:** Kryterium weryfikuje, czy raport, oprócz podania bezwzględnych wartości emisji GHG (w tonach CO2e), przedstawia również wskaźniki intensywności emisji. Pozytywna ocena wymaga ujawnienia co najmniej jednego wskaźnika, który normalizuje emisje w odniesieniu do kluczowego parametru biznesowego (np. przychody, wielkość produkcji, liczba pracowników). Umożliwia to ocenę eko-efektywności operacyjnej spółki oraz ułatwia porównywalność między podmiotami o różnej skali działalności.\n"
        " - **C8 - Wiarygodność celów:** Kryterium ocenia kompletność i wiarygodność strategii redukcji emisji GHG. Pozytywna ocena jest przyznawana wyłącznie raportom, które jednocześnie spełniają dwa warunki: 1) definiują skwantyfikowane, absolutne cele redukcyjne (np. w tonach CO2e) i 2) przedstawiają konkretne działania lub plan, który ma prowadzić do ich osiągnięcia. Każdy przypadek, w którym brakuje jednego z tych dwóch kluczowych komponentów – na przykład obecne są tylko cele intensywności, tylko puste deklaracje celów absolutnych, lub tylko działania bez celów – jest uznawany za strategię niekompletną.\n"
        " - **C9 - Zastosowanie jednostki miary CO2e:** To fundamentalne kryterium weryfikuje, czy raportowane dane dotyczące emisji gazów cieplarnianych są prezentowane w standardowej, zagregowanej jednostce miary – ekwiwalencie dwutlenku węgla (CO2e). Ocena jest dokonywana za pomocą systemu opartego na regułach, który identyfikuje obecność tej jednostki w tekście. Użycie CO2e jest bezwzględnym wymogiem standardu ESRS E1-6, ponieważ świadczy o prawidłowym uwzględnieniu potencjału cieplarnianego (GWP) różnych gazów (np. metanu, podtlenku azotu) i jest warunkiem koniecznym dla agregacji, porównywalności i wiarygodności całego sprawozdania emisyjnego.\n"
    )
    
    analyze_button.click(fn=analyze_report, inputs=pdf_input, outputs=output_df, show_progress='full')

demo.launch()