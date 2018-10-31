// Time this program
#pragma once

#ifndef ONLINE_JUDGE

#include <windows.h>

#include <cstdio>

struct TimeMe {
    ::LARGE_INTEGER qpf;
    ::LARGE_INTEGER qpc;
    TimeMe() {
        ::QueryPerformanceFrequency(&qpf);
        ::QueryPerformanceCounter(&qpc);
    }
    ~TimeMe() {
        ::LARGE_INTEGER now;
        ::QueryPerformanceCounter(&now);
        ::std::fprintf(stderr, "%.6f s\n", (double) (now.QuadPart - qpc.QuadPart) / (double) qpf.QuadPart);
    }
};

TimeMe ImplTimeMe;

#endif
