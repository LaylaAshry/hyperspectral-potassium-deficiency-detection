import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
import glob

import numpy as np
import pandas as pd

from analyzer.loader import load_spectrum
from analyzer.micasense_loader import (load_micasense_capture,
                                        downsample_baseline_to_rededge)
from analyzer.classifier import classify
from analyzer.plotting import plot_comparison
from analyzer.map_display import generate_map
from analyzer.pipeline import (load_baselines_for_rededge,
                                group_into_captures,
                                process_micasense_capture,
                                _detect_band_count)


def resource_path(relative_path):
    """Gets correct path whether running as script or bundled exe."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)


DEFAULT_HEALTHY   = resource_path('baselines/healthy.txt')
DEFAULT_DEFICIENT = resource_path('baselines/deficient.txt')
OUTPUT_LOG        = './results/field_results.csv'
OUTPUT_MAP        = './results/field_map.html'

COLORS = {
    'Healthy':      '#2ecc71',
    'Deficient':    '#e74c3c',
    'Uncertain':    '#f39c12',
    'bg':           '#1a1a2e',
    'panel':        '#16213e',
    'text':         '#eaeaea',
    'subtext':      '#a0a0b0',
    'button':       '#0f3460',
    'button_hover': '#533483',
}


class MangoApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('FAU Farm Owls SpectraSense')
        self.geometry('820x720')
        self.resizable(False, False)
        self.configure(bg=COLORS['bg'])

        self.healthy_path   = tk.StringVar(value=DEFAULT_HEALTHY)
        self.deficient_path = tk.StringVar(value=DEFAULT_DEFICIENT)
        self.capture_files  = []
        self.results        = []
        self.last_map_path  = None

        self._build_ui()
        self._check_baselines()


    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title
        title_frame = tk.Frame(self, bg=COLORS['bg'], pady=16)
        title_frame.pack(fill='x')
        tk.Label(title_frame,
                 text='🌿 FAU Farm Owls SpectraSense',
                 font=('Helvetica', 22, 'bold'),
                 bg=COLORS['bg'], fg=COLORS['text']).pack()
        tk.Label(title_frame,
                 text='MicaSense RedEdge potassium deficiency detector',
                 font=('Helvetica', 10),
                 bg=COLORS['bg'], fg=COLORS['subtext']).pack()

        # Baselines
        baseline_frame = tk.LabelFrame(
            self, text='  Baselines  ',
            bg=COLORS['panel'], fg=COLORS['subtext'],
            font=('Helvetica', 9), padx=12, pady=10)
        baseline_frame.pack(fill='x', padx=20, pady=(0, 10))
        self._file_row(baseline_frame, '🟢 Healthy baseline:',
                       self.healthy_path, row=0)
        self._file_row(baseline_frame, '🔴 Deficient baseline:',
                       self.deficient_path, row=1)

        # Capture files
        capture_frame = tk.LabelFrame(
            self, text='  RedEdge Captures  ',
            bg=COLORS['panel'], fg=COLORS['subtext'],
            font=('Helvetica', 9), padx=12, pady=10)
        capture_frame.pack(fill='x', padx=20, pady=(0, 10))

        btn_row = tk.Frame(capture_frame, bg=COLORS['panel'])
        btn_row.pack(fill='x', pady=(0, 8))
        self._button(btn_row, '📂  Select TIFF Files',
                     self._select_tiffs).pack(
                         side='left', padx=(0, 10))
        self._button(btn_row, '📁  Select Capture Folder',
                     self._select_folder).pack(
                         side='left', padx=(0, 10))
        self._button(btn_row, '✖  Clear',
                     self._clear_captures,
                     small=True).pack(side='left')

        list_frame = tk.Frame(capture_frame, bg=COLORS['panel'])
        list_frame.pack(fill='x')
        sb = tk.Scrollbar(list_frame)
        sb.pack(side='right', fill='y')
        self.file_listbox = tk.Listbox(
            list_frame, height=4,
            bg='#0d1117', fg=COLORS['text'],
            selectbackground=COLORS['button'],
            font=('Courier', 9),
            yscrollcommand=sb.set,
            borderwidth=0, highlightthickness=1,
            highlightcolor=COLORS['button'])
        self.file_listbox.pack(fill='x')
        sb.config(command=self.file_listbox.yview)

        self.file_count_label = tk.Label(
            capture_frame, text='No files selected',
            bg=COLORS['panel'], fg=COLORS['subtext'],
            font=('Helvetica', 9))
        self.file_count_label.pack(anchor='w', pady=(4, 0))

        tk.Label(capture_frame,
                 text='ℹ️  Select all band TIFFs per capture '
                      '(4 bands for RedEdge-3, 5 for RedEdge-MX)',
                 bg=COLORS['panel'], fg=COLORS['subtext'],
                 font=('Helvetica', 8, 'italic')).pack(anchor='w')

        # Run button
        run_frame = tk.Frame(self, bg=COLORS['bg'])
        run_frame.pack(pady=10)
        self.run_btn = self._button(
            run_frame, '▶   Run Analysis',
            self._run_analysis, large=True)
        self.run_btn.pack()

        # Progress
        progress_frame = tk.Frame(self, bg=COLORS['bg'])
        progress_frame.pack(fill='x', padx=20)
        self.progress = ttk.Progressbar(
            progress_frame, mode='determinate', length=780)
        self.progress.pack(fill='x')
        self.progress_label = tk.Label(
            progress_frame, text='',
            bg=COLORS['bg'], fg=COLORS['subtext'],
            font=('Helvetica', 9))
        self.progress_label.pack(anchor='w')

        # Results
        results_frame = tk.LabelFrame(
            self, text='  Results  ',
            bg=COLORS['panel'], fg=COLORS['subtext'],
            font=('Helvetica', 9), padx=12, pady=10)
        results_frame.pack(
            fill='both', expand=True, padx=20, pady=10)

        counts_frame = tk.Frame(results_frame, bg=COLORS['panel'])
        counts_frame.pack(fill='x', pady=(0, 8))
        self.count_healthy   = self._count_badge(
            counts_frame, '✅ Healthy',   COLORS['Healthy'])
        self.count_deficient = self._count_badge(
            counts_frame, '❌ Deficient', COLORS['Deficient'])
        self.count_uncertain = self._count_badge(
            counts_frame, '⚠️ Uncertain', COLORS['Uncertain'])

        list2_frame = tk.Frame(results_frame, bg=COLORS['panel'])
        list2_frame.pack(fill='both', expand=True)
        sb2 = tk.Scrollbar(list2_frame)
        sb2.pack(side='right', fill='y')
        self.results_listbox = tk.Listbox(
            list2_frame,
            bg='#0d1117', fg=COLORS['text'],
            selectbackground=COLORS['button'],
            font=('Courier', 9),
            yscrollcommand=sb2.set,
            borderwidth=0, highlightthickness=1,
            highlightcolor=COLORS['button'])
        self.results_listbox.pack(fill='both', expand=True)
        sb2.config(command=self.results_listbox.yview)
        self.results_listbox.bind(
            '<Double-Button-1>', self._on_result_double_click)

        tk.Label(results_frame,
                 text='Double-click a result to view its '
                      'spectral plot',
                 bg=COLORS['panel'], fg=COLORS['subtext'],
                 font=('Helvetica', 8, 'italic')).pack(
                     anchor='w', pady=(4, 0))

        # Bottom bar
        bottom_frame = tk.Frame(self, bg=COLORS['bg'])
        bottom_frame.pack(fill='x', padx=20, pady=(0, 12))
        self._button(bottom_frame, '🗺  View Field Map',
                     self._open_map,
                     small=True).pack(side='left', padx=(0, 8))
        self._button(bottom_frame, '📊  Open Results CSV',
                     self._open_csv,
                     small=True).pack(side='left', padx=(0, 8))
        self._button(bottom_frame, '🗑  Clear Results',
                     self._clear_results,
                     small=True).pack(side='left')


    # ── Widget helpers ────────────────────────────────────────────

    def _button(self, parent, text, command,
                large=False, small=False):
        size = 13 if large else (9 if small else 11)
        pad  = (16, 10) if large else (10, 6)
        btn = tk.Button(
            parent, text=text, command=command,
            bg=COLORS['button'], fg=COLORS['text'],
            font=('Helvetica', size,
                  'bold' if large else 'normal'),
            relief='flat', cursor='hand2',
            padx=pad[0], pady=pad[1],
            activebackground=COLORS['button_hover'],
            activeforeground=COLORS['text'])
        btn.bind('<Enter>',
                 lambda e: btn.config(
                     bg=COLORS['button_hover']))
        btn.bind('<Leave>',
                 lambda e: btn.config(bg=COLORS['button']))
        return btn

    def _file_row(self, parent, label, var, row):
        tk.Label(parent, text=label, bg=COLORS['panel'],
                 fg=COLORS['text'], font=('Helvetica', 9),
                 width=22, anchor='w').grid(
                     row=row, column=0, sticky='w', pady=3)
        tk.Label(parent, textvariable=var,
                 bg=COLORS['panel'],
                 fg=COLORS['subtext'], font=('Courier', 8),
                 anchor='w').grid(
                     row=row, column=1, sticky='w', padx=8)
        self._button(parent, 'Browse',
                     lambda v=var: self._browse_file(v),
                     small=True).grid(
                         row=row, column=2, padx=4)

    def _count_badge(self, parent, label, color):
        frame = tk.Frame(parent, bg=color, padx=12, pady=6)
        frame.pack(side='left', padx=(0, 8))
        tk.Label(frame, text=label, bg=color, fg='white',
                 font=('Helvetica', 9, 'bold')).pack()
        var = tk.StringVar(value='—')
        tk.Label(frame, textvariable=var, bg=color,
                 fg='white',
                 font=('Helvetica', 18, 'bold')).pack()
        return var


    # ── File selection ────────────────────────────────────────────

    def _browse_file(self, var):
        path = filedialog.askopenfilename(
            filetypes=[('Text files', '*.txt'),
                       ('All files', '*.*')])
        if path:
            var.set(path)
            self._check_baselines()

    def _select_tiffs(self):
        paths = filedialog.askopenfilenames(
            title='Select RedEdge TIFF files',
            filetypes=[('TIFF files', '*.tif *.TIF'),
                       ('All files', '*.*')])
        if paths:
            self._add_captures(list(paths))

    def _select_folder(self):
        folder = filedialog.askdirectory(
            title='Select folder containing RedEdge captures')
        if folder:
            files = (
                glob.glob(os.path.join(folder, '*.tif')) +
                glob.glob(os.path.join(folder, '*.TIF')))
            if files:
                self._add_captures(files)
            else:
                messagebox.showwarning(
                    'No TIFFs found',
                    'No .tif files found in that folder.')

    def _add_captures(self, paths):
        for p in paths:
            if p not in self.capture_files:
                self.capture_files.append(p)
                self.file_listbox.insert(
                    'end', f'  {os.path.basename(p)}')
        n = len(self.capture_files)
        n_bands   = _detect_band_count(self.capture_files)
        captures  = n // n_bands
        remainder = n % n_bands
        status = (f'{n} files selected '
                  f'({captures} complete capture'
                  f'{"s" if captures != 1 else ""}')
        if remainder:
            status += f', {remainder} unmatched'
        status += ')'
        self.file_count_label.config(text=status)

    def _clear_captures(self):
        self.capture_files = []
        self.file_listbox.delete(0, 'end')
        self.file_count_label.config(text='No files selected')


    # ── Baseline check ────────────────────────────────────────────

    def _check_baselines(self):
        missing = []
        for label, var in [('Healthy',   self.healthy_path),
                            ('Deficient', self.deficient_path)]:
            if not os.path.exists(var.get()):
                missing.append(label)
        if missing:
            self.progress_label.config(
                text=f'⚠️  Missing baselines: '
                     f'{", ".join(missing)}',
                fg=COLORS['Uncertain'])
        else:
            self.progress_label.config(
                text='✓ Baselines ready',
                fg=COLORS['Healthy'])


    # ── Analysis ──────────────────────────────────────────────────

    def _run_analysis(self):
        if not self.capture_files:
            messagebox.showwarning(
                'No files',
                'Please select RedEdge TIFF files.')
            return
        self.run_btn.config(state='disabled')
        thread = threading.Thread(
            target=self._analysis_thread, daemon=True)
        thread.start()

    def _analysis_thread(self):
        try:
            self._update_progress(0, 'Loading baselines...')

            n_bands = _detect_band_count(self.capture_files)
            wl_h, r_h, wl_d, r_d = load_baselines_for_rededge(
                self.healthy_path.get(),
                self.deficient_path.get(),
                n_bands)

            captures = group_into_captures(self.capture_files)
            total    = len(captures)

            if total == 0:
                self.after(0, lambda: messagebox.showwarning(
                    'No complete captures',
                    'No complete sets of TIFF files found.\n'
                    'Make sure you select all band files '
                    'per capture.'))
                return

            new_results = []
            for i, (cap_id, files) in enumerate(
                    captures.items()):
                self._update_progress(
                    int((i / total) * 100),
                    f'Processing {i+1}/{total}: {cap_id}')
                result = process_micasense_capture(
                    files, wl_h, r_h, wl_d, r_d, OUTPUT_LOG)
                if result:
                    new_results.append(result)
                    self.results.append(result)

            if any(r.get('lat') for r in new_results):
                self._update_progress(
                    95, 'Generating field map...')
                self.last_map_path = generate_map(
                    new_results, OUTPUT_MAP)

            self._update_progress(
                100,
                f'✓ Done — {len(new_results)} '
                f'capture(s) processed')
            self.after(
                0, lambda: self._display_results(new_results))

        except Exception as e:
            self.after(
                0, lambda: messagebox.showerror(
                    'Error', str(e)))
        finally:
            self.after(
                0, lambda: self.run_btn.config(
                    state='normal'))

    def _update_progress(self, value, text):
        self.after(0,
                   lambda: self.progress.config(value=value))
        self.after(0, lambda: self.progress_label.config(
            text=text, fg=COLORS['text']))


    # ── Results display ───────────────────────────────────────────

    def _display_results(self, new_results):
        counts = {'Healthy': 0, 'Deficient': 0, 'Uncertain': 0}
        for r in new_results:
            label  = r['classification']
            ph     = r['healthy_pct']
            pd_    = r['deficient_pct']
            margin = r['margin_pct']
            icon   = {'Healthy':   '✅',
                      'Deficient': '❌',
                      'Uncertain': '⚠️'}.get(label, '?')
            gps_str = (
                f"📍{r['lat']:.4f},{r['lon']:.4f}"
                if r.get('lat') else '📍 no GPS')

            line = (f'  {icon}  {r["capture_id"]:<20s}  '
                    f'{label:<10s}  '
                    f'H:{ph:5.1f}%  D:{pd_:5.1f}%  '
                    f'margin:{margin:.1f}pp  {gps_str}')

            self.results_listbox.insert('end', line)
            self.results_listbox.itemconfig(
                'end', fg=COLORS[label])
            counts[label] += 1

        self.count_healthy.set(str(counts['Healthy']))
        self.count_deficient.set(str(counts['Deficient']))
        self.count_uncertain.set(str(counts['Uncertain']))

        if self.last_map_path:
            if messagebox.askyesno(
                    'Map ready',
                    'Field map generated. '
                    'Open it now in your browser?'):
                webbrowser.open(
                    f'file://'
                    f'{os.path.abspath(self.last_map_path)}')


    # ── Double-click to plot ──────────────────────────────────────

    def _on_result_double_click(self, event):
        selection = self.results_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx >= len(self.results):
            return

        r = self.results[idx]
        matching = [
            f for f in self.capture_files
            if r['capture_id'] in os.path.basename(f)]
        if not matching:
            return

        try:
            n_bands = _detect_band_count(matching)
            wl_h, r_h, wl_d, r_d = load_baselines_for_rededge(
                self.healthy_path.get(),
                self.deficient_path.get(),
                n_bands)
            wl_s, r_s, _, _ = load_micasense_capture(matching)
            result = classify(wl_s, r_s, wl_h, r_h, wl_d, r_d)
            plot_comparison(wl_s, r_s, wl_h, r_h, wl_d, r_d,
                            r['capture_id'], result)
        except Exception as e:
            messagebox.showerror('Plot error', str(e))


    # ── Map / CSV / Clear ─────────────────────────────────────────

    def _open_map(self):
        if (not self.last_map_path or
                not os.path.exists(self.last_map_path)):
            messagebox.showinfo(
                'No map yet',
                'Run an analysis first. The map is generated '
                'automatically if GPS data is found.')
            return
        webbrowser.open(
            f'file://{os.path.abspath(self.last_map_path)}')

    def _open_csv(self):
        if not os.path.exists(OUTPUT_LOG):
            messagebox.showinfo(
                'No results yet',
                'No results have been saved yet.')
            return
        os.startfile(OUTPUT_LOG)

    def _clear_results(self):
        self.results_listbox.delete(0, 'end')
        self.results        = []
        self.last_map_path  = None
        self.count_healthy.set('—')
        self.count_deficient.set('—')
        self.count_uncertain.set('—')
        self.progress.config(value=0)
        self.progress_label.config(text='')


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    app = MangoApp()
    app.mainloop()