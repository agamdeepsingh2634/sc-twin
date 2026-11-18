"""Keeps Windows awake (no sleep) while this window is open. The screen may still turn off."""
import ctypes, time
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
print("Keeping the laptop awake. Close this window when the runs are finished.")
while True:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    time.sleep(30)
