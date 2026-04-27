import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from analyzer.loader import load_spectrum
from analyzer.calibration import calibrate
from analyzer.classifier import classify
from analyzer.plotting import plot_comparison

def resource_path(relative_path):
    """Gets the correct path whether running as a script or a bundled exe."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)


DEFAULT_HEALTHY   = resource_path('baselines/healthy.txt')
DEFAULT_DEFICIENT = resource_path('baselines/deficient.txt')
DEFAULT_WHITE_REF = resource_path('baselines/white_reference.txt')
OUTPUT_LOG        = './results/field_results.csv'

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

COLORS = {
    'Healthy':   '#2ecc71',
    'Deficient': '#e74c3c',
    'Uncertain': '#f39c12',
    'bg':        '#1a1a2e',
    'panel':     '#16213e',
    'text':      '#eaeaea',
    'subtext':   '#a0a0b0',
    'button':    '#0f3460',
    'button_hover': '#533483',
}


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

class MangoApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('FAU Farm Owls SpectraSense')
        self.geometry('780x680')
        self.resizable(False, False)
        self.configure(bg=COLORS['bg'])

        # State
        self.healthy_path   = tk.StringVar(value=DEFAULT_HEALTHY)
        self.deficient_path = tk.StringVar(value=DEFAULT_DEFICIENT)
        self.white_ref_path = tk.StringVar(value=DEFAULT_WHITE_REF)
        self.sample_files   = []   # list of selected sample file paths
        self.results        = {}   # filename → result dict

        self._build_ui()
        self._check_baselines()


    # ── UI Construction ───────────────────────────────────────────

    def _build_ui(self):
        # ── Title bar ────────────────────────────────────────────
        title_frame = tk.Frame(self, bg=COLORS['bg'], pady=16)
        title_frame.pack(fill='x')
        tk.Label(title_frame, text='🌿 FAU Farm Owls SpectraSense',
                 font=('Helvetica', 22, 'bold'),
                 bg=COLORS['bg'], fg=COLORS['text']).pack()
        tk.Label(title_frame, text='Potassium deficiency detection via hyperspectral imaging',
                 font=('Helvetica', 10),
                 bg=COLORS['bg'], fg=COLORS['subtext']).pack()

        # ── Baseline status panel ─────────────────────────────────
        baseline_frame = tk.LabelFrame(self, text='  Baselines  ',
                                       bg=COLORS['panel'], fg=COLORS['subtext'],
                                       font=('Helvetica', 9),
                                       padx=12, pady=10)
        baseline_frame.pack(fill='x', padx=20, pady=(0, 10))

        self._file_row(baseline_frame, '🟢 Healthy baseline:',
                       self.healthy_path, row=0)
        self._file_row(baseline_frame, '🔴 Deficient baseline:',
                       self.deficient_path, row=1)
        self._file_row(baseline_frame, '⬜ White reference:',
                       self.white_ref_path, row=2)

        # ── Sample files panel ────────────────────────────────────
        sample_frame = tk.LabelFrame(self, text='  Leaf Captures  ',
                                     bg=COLORS['panel'], fg=COLORS['subtext'],
                                     font=('Helvetica', 9),
                                     padx=12, pady=10)
        sample_frame.pack(fill='x', padx=20, pady=(0, 10))

        btn_row = tk.Frame(sample_frame, bg=COLORS['panel'])
        btn_row.pack(fill='x', pady=(0, 8))

        self._button(btn_row, '📂  Select Leaf Files',
                     self._select_samples).pack(side='left', padx=(0, 10))
        self._button(btn_row, '📁  Select Folder',
                     self._select_folder).pack(side='left', padx=(0, 10))
        self._button(btn_row, '✖  Clear',
                     self._clear_samples, small=True).pack(side='left')

        # File list box
        list_frame = tk.Frame(sample_frame, bg=COLORS['panel'])
        list_frame.pack(fill='x')
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        self.file_listbox = tk.Listbox(
            list_frame, height=5,
            bg='#0d1117', fg=COLORS['text'],
            selectbackground=COLORS['button'],
            font=('Courier', 9),
            yscrollcommand=scrollbar.set,
            borderwidth=0, highlightthickness=1,
            highlightcolor=COLORS['button']
        )
        self.file_listbox.pack(fill='x')
        scrollbar.config(command=self.file_listbox.yview)

        self.file_count_label = tk.Label(
            sample_frame, text='No files selected',
            bg=COLORS['panel'], fg=COLORS['subtext'],
            font=('Helvetica', 9)
        )
        self.file_count_label.pack(anchor='w', pady=(4, 0))

        # ── Run button ────────────────────────────────────────────
        run_frame = tk.Frame(self, bg=COLORS['bg'])
        run_frame.pack(pady=10)

        self.run_btn = self._button(
            run_frame, '▶   Run Analysis',
            self._run_analysis, large=True
        )
        self.run_btn.pack()

        # ── Progress bar ──────────────────────────────────────────
        progress_frame = tk.Frame(self, bg=COLORS['bg'])
        progress_frame.pack(fill='x', padx=20)

        self.progress = ttk.Progressbar(progress_frame, mode='determinate',
                                        length=740)
        self.progress.pack(fill='x')

        self.progress_label = tk.Label(
            progress_frame, text='',
            bg=COLORS['bg'], fg=COLORS['subtext'],
            font=('Helvetica', 9)
        )
        self.progress_label.pack(anchor='w')

        # ── Results panel ─────────────────────────────────────────
        results_frame = tk.LabelFrame(self, text='  Results  ',
                                      bg=COLORS['panel'], fg=COLORS['subtext'],
                                      font=('Helvetica', 9),
                                      padx=12, pady=10)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Summary counts
        counts_frame = tk.Frame(results_frame, bg=COLORS['panel'])
        counts_frame.pack(fill='x', pady=(0, 8))

        self.count_healthy   = self._count_badge(counts_frame, '✅ Healthy',   COLORS['Healthy'])
        self.count_deficient = self._count_badge(counts_frame, '❌ Deficient', COLORS['Deficient'])
        self.count_uncertain = self._count_badge(counts_frame, '⚠️ Uncertain', COLORS['Uncertain'])

        # Scrollable results list
        list2_frame = tk.Frame(results_frame, bg=COLORS['panel'])
        list2_frame.pack(fill='both', expand=True)
        scrollbar2 = tk.Scrollbar(list2_frame)
        scrollbar2.pack(side='right', fill='y')
        self.results_listbox = tk.Listbox(
            list2_frame,
            bg='#0d1117', fg=COLORS['text'],
            selectbackground=COLORS['button'],
            font=('Courier', 9),
            yscrollcommand=scrollbar2.set,
            borderwidth=0, highlightthickness=1,
            highlightcolor=COLORS['button']
        )
        self.results_listbox.pack(fill='both', expand=True)
        scrollbar2.config(command=self.results_listbox.yview)
        self.results_listbox.bind('<Double-Button-1>', self._on_result_double_click)

        tk.Label(results_frame,
                 text='Double-click a result to view its spectral plot',
                 bg=COLORS['panel'], fg=COLORS['subtext'],
                 font=('Helvetica', 8, 'italic')).pack(anchor='w', pady=(4, 0))

        # ── Bottom bar ────────────────────────────────────────────
        bottom_frame = tk.Frame(self, bg=COLORS['bg'])
        bottom_frame.pack(fill='x', padx=20, pady=(0, 12))

        self._button(bottom_frame, '📊  Open Results CSV',
                     self._open_csv, small=True).pack(side='left', padx=(0, 8))
        self._button(bottom_frame, '🗑  Clear Results',
                     self._clear_results, small=True).pack(side='left')


    # ── Widget helpers ────────────────────────────────────────────

    def _button(self, parent, text, command, large=False, small=False):
        size = 13 if large else (9 if small else 11)
        pad  = (16, 10) if large else (10, 6)
        btn = tk.Button(
            parent, text=text, command=command,
            bg=COLORS['button'], fg=COLORS['text'],
            font=('Helvetica', size, 'bold' if large else 'normal'),
            relief='flat', cursor='hand2',
            padx=pad[0], pady=pad[1],
            activebackground=COLORS['button_hover'],
            activeforeground=COLORS['text']
        )
        btn.bind('<Enter>', lambda e: btn.config(bg=COLORS['button_hover']))
        btn.bind('<Leave>', lambda e: btn.config(bg=COLORS['button']))
        return btn

    def _file_row(self, parent, label, var, row):
        tk.Label(parent, text=label, bg=COLORS['panel'],
                 fg=COLORS['text'], font=('Helvetica', 9),
                 width=22, anchor='w').grid(row=row, column=0,
                                            sticky='w', pady=3)
        tk.Label(parent, textvariable=var, bg=COLORS['panel'],
                 fg=COLORS['subtext'], font=('Courier', 8),
                 anchor='w').grid(row=row, column=1, sticky='w', padx=8)
        self._button(parent, 'Browse',
                     lambda v=var: self._browse_file(v),
                     small=True).grid(row=row, column=2, padx=4)

    def _count_badge(self, parent, label, color):
        frame = tk.Frame(parent, bg=color, padx=12, pady=6)
        frame.pack(side='left', padx=(0, 8))
        tk.Label(frame, text=label, bg=color, fg='white',
                 font=('Helvetica', 9, 'bold')).pack()
        count_var = tk.StringVar(value='—')
        tk.Label(frame, textvariable=count_var, bg=color, fg='white',
                 font=('Helvetica', 18, 'bold')).pack()
        return count_var


    # ── File selection ────────────────────────────────────────────

    def _browse_file(self, var):
        path = filedialog.askopenfilename(
            filetypes=[('Text files', '*.txt'), ('CSV files', '*.csv'),
                       ('All files', '*.*')]
        )
        if path:
            var.set(path)
            self._check_baselines()

    def _select_samples(self):
        paths = filedialog.askopenfilenames(
            title='Select leaf capture files',
            filetypes=[('Text files', '*.txt'), ('CSV files', '*.csv'),
                       ('All files', '*.*')]
        )
        if paths:
            self._add_samples(list(paths))

    def _select_folder(self):
        folder = filedialog.askdirectory(title='Select folder of leaf captures')
        if folder:
            import glob
            files = (glob.glob(os.path.join(folder, '*.txt')) +
                     glob.glob(os.path.join(folder, '*.csv')))
            if files:
                self._add_samples(files)
            else:
                messagebox.showwarning('No files found',
                                       'No .txt or .csv files found in that folder.')

    def _add_samples(self, paths):
        for p in paths:
            if p not in self.sample_files:
                self.sample_files.append(p)
                self.file_listbox.insert('end', f'  {os.path.basename(p)}')
        n = len(self.sample_files)
        self.file_count_label.config(
            text=f'{n} file{"s" if n != 1 else ""} selected'
        )

    def _clear_samples(self):
        self.sample_files = []
        self.file_listbox.delete(0, 'end')
        self.file_count_label.config(text='No files selected')


    # ── Baseline check ────────────────────────────────────────────

    def _check_baselines(self):
        """Warn if baseline files don't exist yet."""
        missing = []
        for label, var in [('Healthy',   self.healthy_path),
                            ('Deficient', self.deficient_path),
                            ('White ref', self.white_ref_path)]:
            if not os.path.exists(var.get()):
                missing.append(label)
        if missing:
            self.progress_label.config(
                text=f'⚠️  Missing: {", ".join(missing)}',
                fg=COLORS['Uncertain']
            )
        else:
            self.progress_label.config(text='✓ Baselines loaded', fg=COLORS['Healthy'])


    # ── Analysis ──────────────────────────────────────────────────

    def _run_analysis(self):
        if not self.sample_files:
            messagebox.showwarning('No files', 'Please select at least one leaf file.')
            return

        # Run in a thread so the UI doesn't freeze
        self.run_btn.config(state='disabled')
        thread = threading.Thread(target=self._analysis_thread, daemon=True)
        thread.start()

    def _analysis_thread(self):
        try:
            # Load baselines
            self._update_progress(0, 'Loading baselines...')
            wl_h, r_h = load_spectrum(self.healthy_path.get())
            wl_d, r_d = load_spectrum(self.deficient_path.get())
            wl_w, r_w = load_spectrum(self.white_ref_path.get())

            total = len(self.sample_files)
            new_results = {}

            for i, file_path in enumerate(self.sample_files):
                fname = os.path.basename(file_path)
                self._update_progress(
                    int((i / total) * 100),
                    f'Processing {i+1}/{total}: {fname}'
                )

                try:
                    wl_raw, r_raw = load_spectrum(file_path)
                    wl_cal, r_cal = calibrate(wl_raw, r_raw, wl_w, r_w)
                    result = classify(wl_cal, r_cal, wl_h, r_h, wl_d, r_d)
                    new_results[file_path] = result
                    self.results[file_path] = result
                except Exception as e:
                    new_results[file_path] = {'error': str(e)}

            # Save to CSV
            self._save_csv(new_results)

            # Update UI
            self._update_progress(100, f'✓ Done — {total} file(s) processed')
            self.after(0, lambda: self._display_results(new_results))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Error', str(e)))
        finally:
            self.after(0, lambda: self.run_btn.config(state='normal'))

    def _update_progress(self, value, text):
        self.after(0, lambda: self.progress.config(value=value))
        self.after(0, lambda: self.progress_label.config(
            text=text, fg=COLORS['text']))


    # ── Results display ───────────────────────────────────────────

    def _display_results(self, new_results):
        counts = {'Healthy': 0, 'Deficient': 0, 'Uncertain': 0}

        for file_path, result in new_results.items():
            fname = os.path.basename(file_path)

            if 'error' in result:
                self.results_listbox.insert('end', f'  ✗  {fname}  —  Error: {result["error"]}')
                self.results_listbox.itemconfig('end', fg='#888888')
                continue

            label  = result['classification']
            pct_h  = result['healthy_similarity_pct']
            pct_d  = result['deficient_similarity_pct']
            margin = result['margin_pct']
            icon   = {'Healthy': '✅', 'Deficient': '❌', 'Uncertain': '⚠️'}.get(label, '?')

            line = (f'  {icon}  {fname:<35s}  '
                    f'{label:<10s}  '
                    f'H:{pct_h:5.1f}%  D:{pct_d:5.1f}%  '
                    f'margin:{margin:.1f}pp')

            self.results_listbox.insert('end', line)
            self.results_listbox.itemconfig('end', fg=COLORS[label])
            counts[label] += 1

        # Update badge counts
        self.count_healthy.set(str(counts['Healthy']))
        self.count_deficient.set(str(counts['Deficient']))
        self.count_uncertain.set(str(counts['Uncertain']))


    # ── Double-click to plot ──────────────────────────────────────

    def _on_result_double_click(self, event):
        selection = self.results_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx >= len(self.sample_files):
            return

        file_path = self.sample_files[idx]
        result = self.results.get(file_path)
        if not result or 'error' in result:
            return

        try:
            wl_h, r_h = load_spectrum(self.healthy_path.get())
            wl_d, r_d = load_spectrum(self.deficient_path.get())
            wl_w, r_w = load_spectrum(self.white_ref_path.get())
            wl_raw, r_raw = load_spectrum(file_path)
            wl_cal, r_cal = calibrate(wl_raw, r_raw, wl_w, r_w)
            plot_comparison(wl_cal, r_cal, wl_h, r_h, wl_d, r_d,
                            os.path.basename(file_path), result)
        except Exception as e:
            messagebox.showerror('Plot error', str(e))


    # ── CSV ───────────────────────────────────────────────────────

    def _save_csv(self, new_results):
        os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
        rows = []
        for file_path, result in new_results.items():
            if 'error' in result:
                continue
            rows.append({
                'timestamp':                time.strftime('%Y-%m-%d %H:%M:%S'),
                'file':                     os.path.basename(file_path),
                'classification':           result['classification'],
                'healthy_similarity_pct':   result['healthy_similarity_pct'],
                'deficient_similarity_pct': result['deficient_similarity_pct'],
                'margin_pct':               result['margin_pct'],
                'sam_vs_healthy_deg':       result['full_spectrum']['sam_vs_healthy_deg'],
                'sam_vs_deficient_deg':     result['full_spectrum']['sam_vs_deficient_deg'],
                'pearson_vs_healthy':       result['full_spectrum']['pearson_vs_healthy'],
                'pearson_vs_deficient':     result['full_spectrum']['pearson_vs_deficient'],
            })
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_LOG, mode='a',
                      header=not os.path.exists(OUTPUT_LOG),
                      index=False)

    def _open_csv(self):
        if not os.path.exists(OUTPUT_LOG):
            messagebox.showinfo('No results yet',
                                'No results have been saved yet.')
            return
        os.startfile(OUTPUT_LOG)   # Windows — opens in default app (Excel etc.)

    def _clear_results(self):
        self.results_listbox.delete(0, 'end')
        self.results = {}
        self.count_healthy.set('—')
        self.count_deficient.set('—')
        self.count_uncertain.set('—')
        self.progress.config(value=0)
        self.progress_label.config(text='')


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = MangoApp()
    app.mainloop()