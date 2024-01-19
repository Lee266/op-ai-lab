import os

FIXED_DIRECTORY = '/usr/src/ai-lab/src'

class PathFunctions:
    """_summary_
        BasePath: '/usr/src/ai-lab/src'
    """
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

    @staticmethod
    def getFileList(directory: str) -> list:
        """指定されたディレクトリからファイルの一覧を取得します。

        Args:
            directory (str): フォルダのパス

        Returns:
            list: ファイルの一覧
        """
        file_list = []
        directory = PathFunctions.absolutePath(directory)
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_list.append(filename)
        return file_list
