// Single-file distribution; the bundled application is prepared once per build.
// Uses Windows' .NET Framework, never a shell, VBS, or an installed Python.
using System;
using System.Collections;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

internal static class Launcher {
    internal static string Status = "正在检查内置运行环境…";
    internal static int Progress;
    static int exitCode = 1;
    static string logPath;

    [STAThread]
    static int Main(string[] args) {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        using (var view = new LoadingView()) {
            view.Shown += delegate {
                var worker = new Thread(delegate() {
                    try { Run(args, delegate { view.BeginInvoke((Action)delegate { view.Hide(); }); }); }
                    catch (Exception error) {
                        try { File.AppendAllText(logPath, DateTime.Now.ToString("s") + " " + error + Environment.NewLine); } catch { }
                        view.BeginInvoke((Action)delegate {
                            MessageBox.Show(view, "启动失败：" + error.Message + "\n\n详情：" + logPath,
                                            "Royal Lab", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        });
                    }
                    finally { view.BeginInvoke((Action)delegate { view.Close(); }); }
                });
                worker.IsBackground = true;
                worker.Start();
            };
            Application.Run(view);
        }
        return exitCode;
    }

    static void Run(string[] args, Action hide) {
        var clock = Stopwatch.StartNew();
        string outer = Application.ExecutablePath;
        string home = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "RoyalLab");
        Directory.CreateDirectory(home);
        logPath = Path.Combine(home, "launcher.log");
        using (var file = File.Open(outer, FileMode.Open, FileAccess.Read, FileShare.Read)) {
            if (file.Length < 56) throw new InvalidDataException("EXE 文件不完整，请重新复制。");
            file.Position = file.Length - 56;
            var reader = new BinaryReader(file);
            long offset = reader.ReadInt64(), length = reader.ReadInt64();
            byte[] expected = reader.ReadBytes(32);
            if (Encoding.ASCII.GetString(reader.ReadBytes(8)) != "ROYALC01" || offset < 0 ||
                length < 0 || offset > file.Length - 56 || length != file.Length - 56 - offset)
                throw new InvalidDataException("EXE 文件不完整，请重新复制。");
            string id = BitConverter.ToString(expected).Replace("-", "").ToLowerInvariant();
            string cacheBase = Path.Combine(home, "runtime");
            Directory.CreateDirectory(cacheBase);
            string cache = Path.Combine(cacheBase, id);
            // Per-user lock prevents simultaneous first launches from seeing a partial install.
            string user = System.Security.Principal.WindowsIdentity.GetCurrent().User.Value;
            using (var mutex = new Mutex(false, "Local\\RoyalLab-" + user + "-" + id)) {
                Status = "正在准备启动…";
                try { mutex.WaitOne(); } catch (AbandonedMutexException) { }
                try {
                    bool reused = Valid(cache, id);
                    if (!reused) {
                        Status = "首次准备：正在校验内置文件…";
                        using (var payload = new PayloadStream(file, offset, length)) {
                            using (var sha = SHA256.Create()) {
                                if (!sha.ComputeHash(payload).SequenceEqual(expected))
                                    throw new InvalidDataException("内置文件校验失败，请重新复制 EXE。");
                            }
                            payload.Position = 0;
                            Install(payload, cache, id);
                        }
                    }
                    Status = "正在打开 Royal Lab…";
                    var start = new ProcessStartInfo(Path.Combine(cache, "RoyalLab.exe"),
                                                     String.Join(" ", args.Select(Quote))) {
                        UseShellExecute = false, CreateNoWindow = true,
                        WorkingDirectory = Environment.CurrentDirectory
                    };
                    // The outer EXE location defines portable data, not the cache directory.
                    if (String.IsNullOrEmpty(start.EnvironmentVariables["CRBOT_DATA_DIR"])) {
                        string beside = Path.GetDirectoryName(outer);
                        start.EnvironmentVariables["CRBOT_DATA_DIR"] = File.Exists(Path.Combine(beside, "config.json")) ? beside : home;
                    }
                    foreach (string key in start.EnvironmentVariables.Keys.Cast<string>().ToArray()) {
                        if (key.StartsWith("_PYI", StringComparison.OrdinalIgnoreCase) ||
                            new[] { "TCL_LIBRARY", "TK_LIBRARY", "PYTHONHOME", "PYTHONPATH" }.Contains(key.ToUpperInvariant()))
                            start.EnvironmentVariables.Remove(key);
                    }
                    using (var child = Process.Start(start)) {
                        File.AppendAllText(logPath, DateTime.Now.ToString("s") + " cache=" + (reused ? "reuse" : "prepared") +
                            " dispatch_ms=" + clock.ElapsedMilliseconds + " runtime=" + cache + Environment.NewLine);
                        mutex.ReleaseMutex();
                        hide();
                        child.WaitForExit();
                        exitCode = child.ExitCode;
                    }
                } finally {
                    // Release on preparation failure as well; after dispatch it is already released.
                    try { mutex.ReleaseMutex(); } catch (ApplicationException) { }
                }
            }
        }
    }

    internal static string Quote(string arg) {
        // Windows CommandLineToArgvW escaping, including empty args and trailing slashes.
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in arg) {
            if (c == '\\') { slashes++; continue; }
            result.Append('\\', c == '"' ? slashes * 2 + 1 : slashes);
            result.Append(c); slashes = 0;
        }
        result.Append('\\', slashes * 2); result.Append('"');
        return result.ToString();
    }

    static string SafePath(string root, string name) {
        string prefix = Path.GetFullPath(root) + Path.DirectorySeparatorChar;
        string path = Path.GetFullPath(Path.Combine(root, name));
        if (!path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) || name.Contains(":"))
            throw new InvalidDataException("内置文件路径无效。");
        return path;
    }

    static bool Valid(string root, string id) {
        try {
            string[] lines = File.ReadAllLines(Path.Combine(root, ".ready"));
            if (lines.Length < 2 || lines[0] != id || !File.Exists(Path.Combine(root, "RoyalLab.exe"))) return false;
            foreach (string line in lines.Skip(1)) {
                string[] fields = line.Split('\t');
                if (fields.Length != 3) return false;
                var file = new FileInfo(SafePath(root, fields[2]));
                if (!file.Exists || file.Length != Int64.Parse(fields[0]) || file.LastWriteTimeUtc.Ticks != Int64.Parse(fields[1])) return false;
            }
            return true;
        } catch { return false; }
    }

    static void Install(Stream payload, string cache, string id) {
        string staging = cache + ".preparing-" + Guid.NewGuid().ToString("N");
        Directory.CreateDirectory(staging);
        try {
            var manifest = new StringBuilder(id + "\n");
            using (var archive = new ZipArchive(payload, ZipArchiveMode.Read, true)) {
                long total = archive.Entries.Sum(entry => entry.Length), done = 0;
                byte[] buffer = new byte[1024 * 1024];
                foreach (var entry in archive.Entries) {
                    string target = SafePath(staging, entry.FullName);
                    if (entry.FullName.EndsWith("/")) { Directory.CreateDirectory(target); continue; }
                    Directory.CreateDirectory(Path.GetDirectoryName(target));
                    using (var input = entry.Open())
                    using (var output = File.Create(target)) {
                        int count;
                        while ((count = input.Read(buffer, 0, buffer.Length)) > 0) {
                            output.Write(buffer, 0, count); done += count;
                            Progress = (int)(done * 100 / Math.Max(1, total));
                            Status = "首次准备内置运行环境  " + Progress + "%";
                        }
                    }
                    var info = new FileInfo(target);
                    if (info.Length != entry.Length) throw new InvalidDataException("内置文件长度不匹配。");
                    manifest.Append(info.Length).Append('\t').Append(info.LastWriteTimeUtc.Ticks).Append('\t').Append(entry.FullName).Append('\n');
                }
            }
            if (!File.Exists(Path.Combine(staging, "RoyalLab.exe"))) throw new InvalidDataException("缺少内置应用程序。");
            File.WriteAllText(Path.Combine(staging, ".ready"), manifest.ToString());
            string retired = cache + ".replaced-" + Guid.NewGuid().ToString("N");
            bool replacing = Directory.Exists(cache);
            if (replacing) Directory.Move(cache, retired);
            try { Directory.Move(staging, cache); }
            catch {
                if (replacing && !Directory.Exists(cache)) Directory.Move(retired, cache);
                throw;
            }
            // Only this build's immutable runtime; retain locked files for any live instance.
            if (replacing) {
                try { Directory.Delete(retired, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            }
        } finally {
            if (Directory.Exists(staging)) Directory.Delete(staging, true);
        }
    }

    // ZipArchive sees an ordinary ZIP starting at zero, without copying the GB payload.
    sealed class PayloadStream : Stream {
        readonly Stream source;
        readonly long offset, length;
        long position;
        internal PayloadStream(Stream source, long offset, long length) { this.source = source; this.offset = offset; this.length = length; Position = 0; }
        public override bool CanRead { get { return true; } }
        public override bool CanSeek { get { return true; } }
        public override bool CanWrite { get { return false; } }
        public override long Length { get { return length; } }
        public override long Position { get { return position; } set { Seek(value, SeekOrigin.Begin); } }
        public override long Seek(long value, SeekOrigin origin) {
            long next = origin == SeekOrigin.Begin ? value : origin == SeekOrigin.Current ? position + value : length + value;
            if (next < 0 || next > length) throw new IOException("Payload seek out of range");
            source.Position = offset + next; return position = next;
        }
        public override int Read(byte[] buffer, int start, int count) {
            int read = source.Read(buffer, start, (int)Math.Min(count, length - position)); position += read; return read;
        }
        public override void Flush() { }
        public override void SetLength(long value) { throw new NotSupportedException(); }
        public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
    }

    sealed class LoadingView : Form {
        readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
        readonly bool motion = Environment.GetEnvironmentVariable("CRBOT_REDUCED_MOTION") != "1";
        int angle;
        internal LoadingView() {
            Text = "Royal Lab · 正在启动"; ClientSize = new Size(620, 340);
            StartPosition = FormStartPosition.CenterScreen; FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false; MinimizeBox = false; ControlBox = false; DoubleBuffered = true;
            BackColor = Color.FromArgb(11, 17, 32);
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            timer.Interval = 30; timer.Tick += delegate { if (motion) angle = (angle + 6) % 360; Invalidate(); }; timer.Start();
            VisibleChanged += delegate { timer.Enabled = Visible; };
        }
        protected override void OnPaint(PaintEventArgs e) {
            base.OnPaint(e); var g = e.Graphics; g.SmoothingMode = SmoothingMode.AntiAlias;
            using (var line = new Pen(Color.FromArgb(43, 64, 92), 4)) g.DrawEllipse(line, 267, 35, 86, 86);
            using (var line = new Pen(Color.FromArgb(68, 215, 232), 4)) g.DrawArc(line, 267, 35, 86, 86, angle - 90, 100);
            Draw(g, "RL", 26, Color.FromArgb(68, 215, 232), 53);
            Draw(g, "Royal Lab", 28, Color.FromArgb(244, 247, 255), 146);
            Draw(g, Status, 12, Color.FromArgb(166, 183, 206), 211);
            Draw(g, "首次准备完成后，后续启动将直接复用", 10, Color.FromArgb(129, 149, 176), 274);
            if (Progress > 0) {
                using (var brush = new SolidBrush(Color.FromArgb(43, 64, 92))) g.FillRectangle(brush, 130, 253, 360, 3);
                using (var brush = new SolidBrush(Color.FromArgb(68, 215, 232))) g.FillRectangle(brush, 130, 253, 360 * Progress / 100, 3);
            }
        }
        static void Draw(Graphics g, string text, float size, Color color, int y) {
            using (var font = new Font("Microsoft YaHei UI", size))
            using (var brush = new SolidBrush(color))
            using (var format = new StringFormat { Alignment = StringAlignment.Center })
                g.DrawString(text, font, brush, new RectangleF(0, y, 620, 45), format);
        }
        protected override void Dispose(bool disposing) { if (disposing) timer.Dispose(); base.Dispose(disposing); }
    }
}
