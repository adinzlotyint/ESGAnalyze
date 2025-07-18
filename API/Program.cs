using ESGanalyzer.API.Data;
using ESGanalyzer.API.Models.Configuration;
using ESGanalyzer.API.Services;
using ESGanalyzer.API.Services.Analysis;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using System.Text.Json;

namespace ESGanalyzer.API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);
            builder.Configuration.AddUserSecrets<Program>();
            builder.Services.AddControllers();
            builder.Services.AddSwaggerGen();

            var connectionString =
                builder.Configuration.GetConnectionString("DefaultConnection")
                    ?? throw new InvalidOperationException("Connection string"
                    + "'DefaultConnection' not found.");

            builder.Services.AddDbContext<ApplicationDbContext>(options =>
                options.UseSqlServer(connectionString));

            builder.Services.AddIdentity<IdentityUser, IdentityRole>().AddEntityFrameworkStores<ApplicationDbContext>().AddDefaultTokenProviders();

            var jwtKey = builder.Configuration["Jwt:Key"];
            if (string.IsNullOrWhiteSpace(jwtKey))
                throw new InvalidOperationException("JWT key is not configured. Make sure it is set in User Secrets or environment variables.");

            builder.Services.AddAuthentication(options =>
                {
                    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
                    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
                })
                .AddJwtBearer(options => {
                    options.TokenValidationParameters = new TokenValidationParameters {
                        ValidateIssuer = true,
                        ValidateAudience = true,
                        ValidateLifetime = true,
                        ValidateIssuerSigningKey = true,
                        ValidIssuer = builder.Configuration["Jwt:Issuer"],
                        ValidAudience = builder.Configuration["Jwt:Audience"],
                        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey))
                    };
                });

            var configFilePath = Path.Combine(builder.Environment.ContentRootPath, "Data", "regex.json");
            var jsonString = File.ReadAllText(configFilePath);
            var analysisConfig = JsonSerializer.Deserialize<AnalysisConfiguration>(jsonString) ?? throw new 
                    InvalidOperationException("Could not load or parse the regex.json configuration file. Application cannot start.");

            builder.Services.AddSingleton(analysisConfig);

            builder.Services.AddRequestTimeouts();
            builder.Services.AddScoped<IAuthService, AuthService>();
            builder.Services.AddScoped<IParseService, ParseService>();
            builder.Services.AddScoped<IAnalyzer, Analyzer>();
            builder.Services.AddScoped<ICriterias, Criterias>();

            builder.Services.Configure<IdentityOptions>(options =>
            {
                options.User.RequireUniqueEmail = true;
            });


            var app = builder.Build();

            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();

            app.UseAuthentication();
            app.UseAuthorization();

            app.MapControllers();

            app.Run();
        }
    }
}
