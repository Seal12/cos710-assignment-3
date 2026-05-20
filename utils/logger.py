import sys
LOG_LEVEL = 1  # 0 for DEBUG, 1 for INFO

DEBUG = 0
INFO = 1

class Logger:
  @staticmethod
  def debug(message: str):
    if LOG_LEVEL <= DEBUG:
      print(f"[DEBUG] {message}")

  @staticmethod
  def info(message: str):
    if LOG_LEVEL <= INFO:
      # Clear the current line (progress bar) before printing the info log
      sys.stdout.write(f"\r\x1b[K[INFO] {message}\n")
      sys.stdout.flush()

  @staticmethod
  def progress_log(message: str):
    # \r goes to start of line, \x1b[K clears to the end
    sys.stdout.write(f"\r\x1b[K{message}")
    sys.stdout.flush()

# Example usage/testing
if __name__ == "__main__":
  Logger.info("This is an info message")
  Logger.debug("This is a debug message")
  
  print("\nSwitching to INFO level...")
  LOG_LEVEL = 1
  Logger.info("This info message should show")
  Logger.debug("This debug message should NOT show")
