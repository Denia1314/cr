Option Explicit

Dim shell, files, root, launcher, command, result
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")

root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
launcher = root & "\start_ui.bat"
command = "cmd.exe /d /c """ & launcher & """"
On Error Resume Next
result = shell.Run(command, 0, False)
If Err.Number <> 0 Then
    MsgBox "Unable to start the Royal Lab launcher." & vbCrLf & vbCrLf & _
        "Project directory: " & root, vbCritical, "Royal Lab Startup Error"
End If
On Error GoTo 0
