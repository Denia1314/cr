Option Explicit

Dim shell, files, root, launcher, command
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")

root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
launcher = root & "\start_ui.bat"
command = "cmd.exe /d /c """ & launcher & """"
shell.Run command, 0, False
