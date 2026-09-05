Option Explicit

Dim shell, files, root, pythonw, command
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")

root = files.GetParentFolderName(WScript.ScriptFullName)
pythonw = shell.ExpandEnvironmentStrings("%USERPROFILE%") & "\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"

If Not files.FileExists(pythonw) Then
    pythonw = "pythonw.exe"
End If

shell.CurrentDirectory = root
command = """" & pythonw & """ """ & root & "\launch_ui.pyw"""
shell.Run command, 0, False
