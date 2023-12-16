class Time:
    def __init__(self) -> None:
        print("Active Time instance created")

    @staticmethod
    def convert_seconds(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return hours, minutes, seconds

    def convert_time_text(self, seconds):
        hours, minutes, second = self.convert_seconds(seconds)
        time_text = ""
        if hours > 0:
          time_text += f"{hours:.0f}h:"
        if minutes > 0:
          time_text += f"{minutes:.0f}m:"
        time_text += f"{second:.4f}s"
        return time_text
