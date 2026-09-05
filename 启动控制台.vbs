Option Explicit

Dim shell, files, root, localPythonw, bundledPythonw, pythonw, command, result
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")

root = files.GetParentFolderName(WScript.ScriptFullName)
localPythonw = root & "\.venv\Scripts\pythonw.exe"
bundledPythonw = shell.ExpandEnvironmentStrings("%USERPROFILE%") & "\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"

If files.FileExists(localPythonw) Then
    pythonw = localPythonw
ElseIf files.FileExists(bundledPythonw) Then
    pythonw = bundledPythonw
Else
    pythonw = "pythonw.exe"
End If

shell.CurrentDirectory = root
command = """" & pythonw & """ """ & root & "\launch_ui.pyw"""
On Error Resume Next
result = shell.Run(command, 0, False)
If Err.Number <> 0 Then
    MsgBox "Unable to start Python 3.10 or newer." & vbCrLf & vbCrLf & _
        "Project directory: " & root, vbCritical, "Royal Lab Startup Error"
End If
On Error GoTo 0
