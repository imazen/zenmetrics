#include "VshipAPI.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    Vship_Version v = Vship_GetVersion();
    printf("vship %d.%d.%d  backend=%d (0=HIP, 1=CUDA, 2=Vulkan)\n", v.major, v.minor, v.minorMinor, v.backend);

    int n = 0;
    Vship_Exception e = Vship_GetDeviceCount(&n);
    printf("device count: %d (err=%d)\n", n, e);

    for (int i = 0; i < n; i++) {
        Vship_DeviceInfo info;
        e = Vship_GetDeviceInfo(&info, i);
        printf("  [%d] %s  vram=%llu MB  cu=%d  warpSize=%d  integrated=%d (err=%d)\n",
            i, info.name, (unsigned long long)(info.VRAMSize/1024/1024),
            info.MultiProcessorCount, info.WarpSize, info.integrated, e);
    }

    e = Vship_GPUFullCheck(0);
    char msg[1024] = {0};
    Vship_GetErrorMessage(e, msg, sizeof(msg));
    printf("GPUFullCheck(0) = %d (%s)\n", e, msg);
    return 0;
}
