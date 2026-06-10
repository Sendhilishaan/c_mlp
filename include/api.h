#ifndef API_H
#define API_H

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void* api_linear_create(int in_features, int out_features, const char* activation);

EXPORT void* api_mlp_create(void** layers, int n_layers);

EXPORT void api_mlp_run(void* mlp, double* X, double* y, int rows, int cols, int epochs, double lr);

#endif