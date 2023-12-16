import os

FIXED_DIRECTORY = '/usr/src/ai-lab/src'

class PathFunctions:
    def __init__(self) -> None:
        print("Active PathFunctions instance created")

    @staticmethod
    def absolutePath(targetPath: str) -> str:
        """与えられた相対パスに基づいて絶対パスを生成します。

        Args:
            targetPath (str): ファイルパスの先頭に/はいりませんが、ある場合は削除されます。

        Returns:
            str: 生成された絶対パス
        """
        absolute_path = os.path.join(FIXED_DIRECTORY, targetPath.lstrip('/'))

        return absolute_path
