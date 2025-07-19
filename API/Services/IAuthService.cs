using ESGanalyzer.Shared.DTOs;

namespace ESGanalyzer.API.Services {
    public interface IAuthService {
        Task<string> RegisterAsync(RegisterRequest request);
        Task<string> LoginAsync(LoginRequest request);
    }
}
