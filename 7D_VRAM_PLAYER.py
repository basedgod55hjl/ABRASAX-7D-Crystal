# 7D VRAM PLAYER v1.0
# OSIRISBLXCK — ABRASAX God OS
# PRIMAL_HEX: 4f5349524953424c58434b | φ=1.618033988749895
# A custom media player wrapping VLC with 7D VRAM enhancement technology

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import os
import time
import threading
import json
import re
import html
import sys
from pathlib import Path
from datetime import datetime, timedelta
import queue

# ============================================
# CONFIGURATION
# ============================================
VLC_PATH = r"C:\Users\BASEDGOD\Desktop\ABRASAX\vlc\bin\vlc.exe"
PRIMAL_HEX = "4f5349524953424c58434b"
PHI = 1.618033988749895
K = 0.6180339887498949

VLC_FLAGS = [
    '--avcodec-hw=any',
    '--avcodec-dr',
    '--avcodec-fast',
    '--avcodec-threads=16',
    '--vout=direct3d11',
    '--d3d11-upscale-mode=linear',
    '--d3d11-hdr-mode=auto',
    '--direct3d11-hw-blending',
    '--file-caching=1500',
    '--network-caching=3000',
    '--live-caching=500',
    '--disc-caching=500',
    '--video-filter=sharpen,deinterlace,adjust',
    '--sharpen-sigma=0.5',
    '--deinterlace=1',
    '--deinterlace-mode=yadif',
    '--postproc-q=6',
    '--contrast=1.15',
    '--brightness=1.05',
    '--saturation=1.1',
    '--gamma=1.0',
    '--hue=0',
    '--audio-filter=spatializer,compressor,equalizer,stereo_widen',
    '--spatializer-roomsize=0.85',
    '--spatializer-width=1.0',
    '--spatializer-wet=0.4',
    '--spatializer-dry=0.6',
    '--spatializer-damp=0.2',
    '--compressor-rms-peak=0',
    '--compressor-attack=10',
    '--compressor-release=150',
    '--compressor-threshold=-18',
    '--compressor-ratio=4',
    '--compressor-knee=5',
    '--compressor-makeup-gain=6',
    '--equalizer-preset=fullbass',
    '--equalizer-preamp=18.0',
    '--audio-time-stretch',
    '--audio-replay-gain-mode=track',
    '--playlist-autostart',
    '--loop',
    '--repeat',
    '--no-video-title-show',
    '--video-on-top',
    '--audio-desync=0',
]


class SEVEN_D_VRAM_Player:
    """7D VRAM Player — Custom media player with GPU monitoring"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"7D VRAM PLAYER — OSIRISBLXCK [{PRIMAL_HEX[:16]}...]")
        self.root.geometry("1280x800")
        self.root.configure(bg='#0a0a0f')
        
        # State
        self.vlc_process = None
        self.monitoring = False
        self.playlist_path = ""
        self.current_playlist = []
        self.video_stats = {}
        self.gpu_data = queue.Queue()
        
        # Try to set dark title bar
        try:
            from ctypes import windll, c_int, byref
            windll.dwmapi.DwmSetWindowAttribute(
                windll.kernel32.GetParent(self.root.winfo_id()),
                20, byref(c_int(2)), 4
            )
        except:
            pass
        
        self._build_ui()
        self._load_default_playlist()
        self._start_gpu_monitor()
        
    def _build_ui(self):
        """Build the complete UI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Define custom colors
        BG_DARK = '#0a0a0f'
        BG_MID = '#12121a'
        BG_CARD = '#1a1a2e'
        ACCENT = '#00ff88'
        ACCENT2 = '#ff0066'
        TEXT = '#e0e0e0'
        TEXT_DIM = '#666688'
        
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ========== TOP BAR ==========
        top = tk.Frame(main_frame, bg=BG_MID, height=60)
        top.pack(fill=tk.X, pady=(0, 10))
        top.pack_propagate(False)
        
        # Logo / Title
        title_frame = tk.Frame(top, bg=BG_MID)
        title_frame.pack(side=tk.LEFT, padx=15, pady=5)
        
        tk.Label(title_frame, text="⬡ 7D VRAM", font=('Consolas', 18, 'bold'),
                fg=ACCENT, bg=BG_MID).pack(anchor='w')
        tk.Label(title_frame, text=f"PRIMAL: {PRIMAL_HEX} | φ={PHI:.6f}",
                font=('Consolas', 8), fg=TEXT_DIM, bg=BG_MID).pack(anchor='w')
        
        # GPU Status in top bar
        self.gpu_frame = tk.Frame(top, bg=BG_MID)
        self.gpu_frame.pack(side=tk.RIGHT, padx=15)
        
        self.gpu_labels = {}
        gpu_info = [
            ('gpu_temp', 'TEMP', '--°C'),
            ('gpu_util', 'GPU', '--%'),
            ('gpu_vram', 'VRAM', '--/6144'),
            ('gpu_pwr', 'PWR', '--W'),
        ]
        for key, label, default in gpu_info:
            f = tk.Frame(self.gpu_frame, bg=BG_MID)
            f.pack(side=tk.LEFT, padx=5)
            tk.Label(f, text=label, font=('Consolas', 7), fg=TEXT_DIM, bg=BG_MID).pack()
            self.gpu_labels[key] = tk.Label(f, text=default, font=('Consolas', 10, 'bold'),
                                           fg=ACCENT, bg=BG_MID)
            self.gpu_labels[key].pack()
        
        # Separator
        sep = tk.Frame(main_frame, bg=ACCENT, height=1)
        sep.pack(fill=tk.X, pady=5)
        
        # ========== MAIN CONTENT ==========
        content = tk.Frame(main_frame, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True)
        
        # LEFT PANEL — 7D VRAM Visualizer + Controls
        left = tk.Frame(content, bg=BG_DARK, width=400)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        left.pack_propagate(False)
        
        # 7D VRAM Visualizer Canvas
        viz_frame = tk.LabelFrame(left, text="⚡ 7D VRAM VISUALIZER", 
                                  font=('Consolas', 9, 'bold'),
                                  fg=ACCENT, bg=BG_CARD, bd=1, relief='solid')
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.viz_canvas = tk.Canvas(viz_frame, bg='#0d0d1a', highlightthickness=0,
                                    height=200)
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._draw_vram_visualization()
        
        # Controls
        ctrl_frame = tk.LabelFrame(left, text="🎮 CONTROLS",
                                   font=('Consolas', 9, 'bold'),
                                   fg=ACCENT, bg=BG_CARD, bd=1, relief='solid')
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Launch controls
        btn_frame = tk.Frame(ctrl_frame, bg=BG_CARD)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.launch_btn = tk.Button(btn_frame, text="🚀 LAUNCH VLC (7D VRAM)",
                                   command=self._toggle_vlc,
                                   font=('Consolas', 10, 'bold'),
                                   bg='#00aa55', fg='white', 
                                   activebackground='#00cc66',
                                   bd=0, padx=15, pady=8, cursor='hand2')
        self.launch_btn.pack(fill=tk.X, pady=2)
        
        tk.Button(btn_frame, text="📂 LOAD PLAYLIST",
                 command=self._load_playlist,
                 font=('Consolas', 9), bg='#1a2a4e', fg=TEXT,
                 activebackground='#2a3a5e', bd=0, padx=10, pady=5,
                 cursor='hand2').pack(fill=tk.X, pady=2)
        
        # Enhancement toggles
        toggle_frame = tk.Frame(ctrl_frame, bg=BG_CARD)
        toggle_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.enhancements = {}
        toggles = [
            ('hw_decode', '⚡ HW Decode', True),
            ('sharpen', '🔪 Sharpen 0.5', True),
            ('deinterlace', '🔄 Yadif', True),
            ('adjust', '🎨 Adjust', True),
            ('spatializer', '🌊 Spatializer', True),
            ('compressor', '🔊 Compressor', True),
            ('equalizer', '🎵 EQ FullBass', True),
        ]
        
        for key, label, default in toggles:
            var = tk.BooleanVar(value=default)
            self.enhancements[key] = var
            tk.Checkbutton(toggle_frame, text=label, variable=var,
                          font=('Consolas', 8), fg=TEXT_DIM, bg=BG_CARD,
                          selectcolor='#0a0a0f', activebackground=BG_CARD,
                          fg_active=ACCENT).pack(anchor='w')
        
        # RIGHT PANEL — Playlist + Stats
        right = tk.Frame(content, bg=BG_DARK)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Stats bar
        self.stats_frame = tk.Frame(right, bg=BG_MID, height=35)
        self.stats_frame.pack(fill=tk.X, pady=(0, 5))
        self.stats_frame.pack_propagate(False)
        
        self.stats_labels = {}
        for key, label, default in [
            ('count', 'VIDEOS', '--'),
            ('total_time', 'TOTAL TIME', '--'),
            ('total_size', 'TOTAL SIZE', '--'),
            ('avg_len', 'AVG LEN', '--'),
        ]:
            f = tk.Frame(self.stats_frame, bg=BG_MID)
            f.pack(side=tk.LEFT, padx=10, pady=2)
            tk.Label(f, text=label, font=('Consolas', 7), fg=TEXT_DIM, bg=BG_MID).pack()
            lbl = tk.Label(f, text=default, font=('Consolas', 9, 'bold'),
                          fg=ACCENT, bg=BG_MID)
            lbl.pack()
            self.stats_labels[key] = lbl
        
        # Playlist tree
        tree_frame = tk.LabelFrame(right, text="📋 PLAYLIST (sorted by duration)",
                                   font=('Consolas', 9, 'bold'),
                                   fg=ACCENT, bg=BG_CARD, bd=1, relief='solid')
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('#', 'Duration', 'Size', 'Resolution', 'Filename')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                 height=15)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=60 if col in ('#',) else 80 if col in ('Duration','Size','Resolution') else 350)
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Bottom status bar
        status_bar = tk.Frame(main_frame, bg=BG_MID, height=25)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        status_bar.pack_propagate(False)
        
        self.status = tk.Label(status_bar, text="⏳ Initializing...",
                              font=('Consolas', 8), fg=TEXT_DIM, bg=BG_MID)
        self.status.pack(side=tk.LEFT, padx=10)
        
        tk.Label(status_bar, text=f"φ×K = {PHI*K:.10f}",
                font=('Consolas', 8), fg=TEXT_DIM, bg=BG_MID).pack(side=tk.RIGHT, padx=10)
        
    def _draw_vram_visualization(self):
        """Draw the 7D VRAM visualization on canvas"""
        c = self.viz_canvas
        c.delete('all')
        w = c.winfo_width() or 380
        h = c.winfo_height() or 180
        
        # Draw concentric hexagons (7D concept)
        colors = ['#00ff88', '#00cc77', '#009966', '#006655', '#004433', '#002211', '#001108']
        center_x, center_y = w//2, h//2
        
        for i in range(7):
            size = 60 - i * 7
            if size < 5: break
            alpha = 1.0 - (i * 0.12)
            color = colors[i] if i < len(colors) else '#00ff88'
            
            # Draw hexagon
            points = []
            for j in range(6):
                angle = (j * 60 - 30) * 3.14159 / 180
                px = center_x + size * 1.5 * (1 + i * 0.05) * __import__('math').cos(angle)
                py = center_y + size * 1.5 * (1 + i * 0.05) * __import__('math').sin(angle)
                points.extend([px, py])
            
            c.create_polygon(points, outline=color, fill='', width=2-0.2*i)
        
        # Center label
        c.create_text(center_x, center_y-10, text=f"φ = {PHI:.6f}",
                     font=('Consolas', 10, 'bold'), fill='#00ff88')
        c.create_text(center_x, center_y+15, text=f"φ×K = {PHI*K:.10f}",
                     font=('Consolas', 7), fill='#006644')
        
        # VRAM bars (animated placeholder)
        bar_y = h - 30
        for i in range(12):
            bx = 20 + i * 30
            bh = 10 + (i * 3 % 15)
            c.create_rectangle(bx, bar_y - bh, bx + 15, bar_y,
                              fill='#00ff88' if i < 6 else '#006644',
                              outline='')
        
        # Schedule redraw for animation
        self.root.after(2000, self._draw_vram_visualization)
        
    def _load_default_playlist(self):
        """Load the default playlist"""
        default_playlist = r"C:\Users\BASEDGOD\Desktop\ALL_VIDEOS_BY_DURATION_DEDUPED.xspf"
        alt_playlist = r"C:\Users\BASEDGOD\Desktop\ALL_VIDEOS_BY_DURATION.xspf"
        
        path = default_playlist if os.path.exists(default_playlist) else \
               alt_playlist if os.path.exists(alt_playlist) else None
        
        if path:
            self._parse_and_load_playlist(path)
    
    def _parse_and_load_playlist(self, path):
        """Parse an XSPF playlist and load into tree"""
        self.playlist_path = path
        self.status.config(text=f"Loading playlist: {os.path.basename(path)}...")
        self.root.update()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            locations = re.findall(r'<location>file:///(.+?)</location>', content)
            self.current_playlist = [html.unescape(l) for l in locations]
            
            # Load stats from JSON if available
            stats_path = r"C:\Users\BASEDGOD\Desktop\ABRASAX\video_scan_results.json"
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    all_stats = json.load(f)
                self.video_stats = {v['path']: v for v in all_stats}
            
            # Populate tree
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            total_duration = 0
            total_size = 0
            heights = set()
            
            for i, vpath in enumerate(self.current_playlist):
                if not os.path.exists(vpath):
                    continue
                
                fsize = os.path.getsize(vpath)
                total_size += fsize
                
                # Get duration from stats
                dur = self.video_stats.get(vpath, {}).get('duration', 0)
                total_duration += dur
                
                h = int(dur // 3600)
                m = int((dur % 3600) // 60)
                s = int(dur % 60)
                dur_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
                
                size_str = f"{fsize/(1024*1024):.0f}MB"
                w = self.video_stats.get(vpath, {}).get('width', 0)
                hgt = self.video_stats.get(vpath, {}).get('height', 0)
                res_str = f"{w}x{hgt}" if w else '???'
                if hgt: heights.add(hgt)
                
                fname = os.path.basename(vpath)[:60]
                
                self.tree.insert('', 'end', values=(i+1, dur_str, size_str, res_str, fname))
                
                # Update in batches
                if i % 50 == 0:
                    self.root.update()
            
            # Update stats
            hrs = total_duration / 3600
            self.stats_labels['count'].config(text=str(len(self.current_playlist)))
            self.stats_labels['total_time'].config(text=f"{hrs:.1f}h")
            self.stats_labels['total_size'].config(text=f"{total_size/(1024**3):.1f}GB")
            avg = total_duration / len(self.current_playlist) / 60 if self.current_playlist else 0
            self.stats_labels['avg_len'].config(text=f"{avg:.1f}min")
            
            self.status.config(text=f"✅ Loaded {len(self.current_playlist)} videos | {hrs:.1f}h total")
            
        except Exception as e:
            self.status.config(text=f"❌ Error: {e}")
    
    def _load_playlist(self):
        """Load a playlist file"""
        path = filedialog.askopenfilename(
            title="Select Playlist",
            filetypes=[("XSPF Playlists", "*.xspf"), ("All Files", "*.*")]
        )
        if path:
            self._parse_and_load_playlist(path)
    
    def _toggle_vlc(self):
        """Launch or kill VLC"""
        if self.vlc_process and self.vlc_process.poll() is None:
            # Kill
            self.vlc_process.kill()
            self.vlc_process = None
            self.launch_btn.config(text="🚀 LAUNCH VLC (7D VRAM)", bg='#00aa55')
            self.status.config(text="VLC stopped")
            return
        
        # Launch
        if not self.playlist_path or not os.path.exists(self.playlist_path):
            messagebox.showerror("Error", "No playlist loaded!")
            return
        
        # Build args based on toggles
        args = [VLC_PATH, self.playlist_path]
        
        # Add active enhancements
        if self.enhancements['hw_decode'].get():
            args.extend(['--avcodec-hw=any', '--avcodec-dr', '--avcodec-fast', '--avcodec-threads=16'])
            args.extend(['--vout=direct3d11', '--d3d11-upscale-mode=linear', '--d3d11-hdr-mode=auto', '--direct3d11-hw-blending'])
        
        args.extend(['--file-caching=1500', '--network-caching=3000', '--live-caching=500', '--disc-caching=500'])
        
        filters = []
        if self.enhancements['sharpen'].get(): filters.append('sharpen')
        if self.enhancements['deinterlace'].get(): filters.append('deinterlace')
        if self.enhancements['adjust'].get(): filters.append('adjust')
        if filters:
            args.append(f'--video-filter={",".join(filters)}')
            if 'sharpen' in filters: args.append('--sharpen-sigma=0.5')
            if 'deinterlace' in filters: args.extend(['--deinterlace=1', '--deinterlace-mode=yadif'])
            if 'adjust' in filters: args.extend(['--contrast=1.15', '--brightness=1.05', '--saturation=1.1', '--gamma=1.0'])
        
        args.append('--postproc-q=6')
        
        audio_filters = []
        if self.enhancements['spatializer'].get(): audio_filters.append('spatializer')
        if self.enhancements['compressor'].get(): audio_filters.append('compressor')
        if self.enhancements['equalizer'].get(): audio_filters.append('equalizer')
        audio_filters.append('stereo_widen')
        if audio_filters:
            args.append(f'--audio-filter={",".join(audio_filters)}')
            if 'spatializer' in audio_filters: args.extend(['--spatializer-roomsize=0.85', '--spatializer-width=1.0'])
            if 'compressor' in audio_filters: args.extend(['--compressor-ratio=4', '--compressor-threshold=-18', '--compressor-makeup-gain=6'])
            if 'equalizer' in audio_filters: args.extend(['--equalizer-preset=fullbass', '--equalizer-preamp=18.0'])
        
        args.extend([
            '--playlist-autostart', '--loop', '--repeat',
            '--no-video-title-show', '--video-on-top'
        ])
        
        try:
            self.vlc_process = subprocess.Popen(args, shell=False,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.launch_btn.config(text="⏹ STOP VLC", bg='#cc3333')
            self.status.config(text=f"✅ VLC launched (PID: {self.vlc_process.pid})")
        except Exception as e:
            self.status.config(text=f"❌ Launch failed: {e}")
    
    def _start_gpu_monitor(self):
        """Start GPU monitoring thread"""
        self.monitoring = True
        thread = threading.Thread(target=self._gpu_monitor_loop, daemon=True)
        thread.start()
    
    def _gpu_monitor_loop(self):
        """Monitor GPU stats"""
        while self.monitoring:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    parts = result.stdout.strip().split(', ')
                    if len(parts) >= 4:
                        data = {
                            'temp': parts[0],
                            'util': parts[1],
                            'vram': parts[2],
                            'pwr': parts[3],
                        }
                        self.gpu_data.put(data)
            except:
                pass
            
            # Update VLC status
            if self.vlc_process:
                if self.vlc_process.poll() is not None:
                    self.vlc_process = None
                    self.root.after(0, lambda: self.launch_btn.config(
                        text="🚀 LAUNCH VLC (7D VRAM)", bg='#00aa55'))
            
            time.sleep(2.5)
    
    def _update_gpu_display(self):
        """Update GPU display from queue"""
        try:
            while True:
                data = self.gpu_data.get_nowait()
                self.gpu_labels['gpu_temp'].config(text=f"{data['temp']}°C")
                self.gpu_labels['gpu_util'].config(text=f"{data['util']}%")
                self.gpu_labels['gpu_vram'].config(text=f"{data['vram']}/6144")
                self.gpu_labels['gpu_pwr'].config(text=f"{data['pwr']}W")
                
                # Color code based on utilization
                util = int(data['util'])
                color = '#00ff88' if util < 30 else '#ffaa00' if util < 70 else '#ff0066'
                self.gpu_labels['gpu_util'].config(fg=color)
        except queue.Empty:
            pass
        
        self.root.after(2500, self._update_gpu_display)


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SEVEN_D_VRAM_Player(root)
    root.after(1000, app._update_gpu_display)
    root.mainloop()

