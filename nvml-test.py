from pynvml import nvmlDeviceResetNvLinkUtilizationCounter , nvmlInit, nvmlDeviceGetNvLinkUtilizationCounter, link
nvmlInit()


nvmlDeviceGetNvLinkUtilizationCounter(0,1,1)
