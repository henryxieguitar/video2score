#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video-sheet-gui.py — 视频转乐谱 GUI 界面 v2.0
video-sheet-extract.py 的 Tkinter 图形界面封装，通过子进程调用核心引擎。

综合1.0版GUI功能 + 1.1版进程终止
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import cv2
from PIL import Image, ImageTk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_PY = os.path.join(SCRIPT_DIR, 'video-sheet-extract.py')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'video-sheet-config.json')


def sec2hms(sec):
    sec = int(sec)
    return f'{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


# =====================================================================
# 子窗口：框选裁剪区域
# =====================================================================
class CropSelectorWindow(tk.Toplevel):
    def __init__(self, master, video, on_confirm):
        super().__init__(master)
        self.title('框选裁剪区域')
        self.on_confirm = on_confirm
        self.cap = cv2.VideoCapture(video)
        self.total = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        self.scale = 1.0
        self.rect = None
        self.y0 = self.y1 = 0

        self.slider = tk.Scale(self, from_=0, to=self.total - 1,
                               orient='horizontal', label='帧位置',
                               command=self.show_frame)
        self.slider.pack(fill='x', padx=8)
        self.canvas = tk.Canvas(self, width=800, height=450, bg='black')
        self.canvas.pack(padx=8, pady=4)
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)

        bar = ttk.Frame(self); bar.pack(fill='x', padx=8, pady=6)
        ttk.Label(bar, text='Y起始:').pack(side='left')
        self.var_y0 = tk.StringVar(value='0')
        ttk.Entry(bar, textvariable=self.var_y0, width=8).pack(side='left')
        ttk.Label(bar, text='Y结束:').pack(side='left', padx=(10, 0))
        self.var_y1 = tk.StringVar(value=str(self.fh))
        ttk.Entry(bar, textvariable=self.var_y1, width=8).pack(side='left')
        ttk.Button(bar, text='确认', command=self.confirm).pack(side='right')

        self.slider.set(self.total // 2)
        self.show_frame(self.total // 2)

    def show_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(idx)))
        ok, frame = self.cap.read()
        if not ok:
            return
        h, w = frame.shape[:2]
        self.scale = min(800 / w, 450 / h)
        disp = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.draw_rect()

    def on_press(self, e):
        self.y0 = e.y
        self.y1 = e.y

    def on_drag(self, e):
        self.y1 = e.y
        self.var_y0.set(str(int(min(self.y0, self.y1) / self.scale)))
        self.var_y1.set(str(int(max(self.y0, self.y1) / self.scale)))
        self.draw_rect()

    def draw_rect(self):
        self.canvas.delete('rect')
        try:
            a = int(self.var_y0.get()) * self.scale
            b = int(self.var_y1.get()) * self.scale
            self.canvas.create_rectangle(0, a, 800, b, outline='red',
                                         width=2, tags='rect')
        except ValueError:
            pass

    def confirm(self):
        try:
            self.on_confirm(int(self.var_y0.get()), int(self.var_y1.get()))
        except ValueError:
            messagebox.showerror('错误', 'Y 坐标必须为整数', parent=self)
            return
        self.cap.release()
        self.destroy()


# =====================================================================
# 子窗口：时间范围选择（带播放）
# =====================================================================
class TimeSelectorWindow(tk.Toplevel):
    def __init__(self, master, video, on_confirm):
        super().__init__(master)
        self.title('选择时间范围')
        self.on_confirm = on_confirm
        self.cap = cv2.VideoCapture(video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.total = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.playing = False
        self.t_start, self.t_end = '', ''

        self.canvas = tk.Canvas(self, width=800, height=450, bg='black')
        self.canvas.pack(padx=8, pady=4)
        self.slider = tk.Scale(self, from_=0, to=self.total - 1,
                               orient='horizontal', label='时间轴',
                               command=self.seek)
        self.slider.pack(fill='x', padx=8)

        bar = ttk.Frame(self); bar.pack(fill='x', padx=8, pady=6)
        ttk.Button(bar, text='播放', command=self.play).pack(side='left')
        ttk.Button(bar, text='暂停', command=self.pause).pack(side='left')
        ttk.Button(bar, text='停止', command=self.stop).pack(side='left')
        self.lbl = ttk.Label(bar, text='00:00:00')
        self.lbl.pack(side='left', padx=10)
        ttk.Button(bar, text='设为起始', command=self.set_start).pack(side='left', padx=4)
        ttk.Button(bar, text='设为结束', command=self.set_end).pack(side='left')
        self.lbl_range = ttk.Label(bar, text='起始: --  结束: --')
        self.lbl_range.pack(side='left', padx=10)
        ttk.Button(bar, text='确认', command=self.confirm).pack(side='right')
        self.seek(0)

    def cur_time(self):
        return self.slider.get() / self.fps

    def seek(self, idx):
        if self.playing:
            return
        self.render(int(float(idx)))

    def render(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        h, w = frame.shape[:2]
        s = min(800 / w, 450 / h)
        disp = cv2.cvtColor(cv2.resize(frame, (int(w * s), int(h * s))),
                            cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.lbl.config(text=sec2hms(idx / self.fps))

    def play(self):
        if self.playing:
            return
        self.playing = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.playing:
            idx = self.slider.get() + max(1, int(self.fps // 5))
            if idx >= self.total:
                self.playing = False
                break
            self.slider.set(idx)
            try:
                self.render(idx)
            except tk.TclError:
                break
            time.sleep(0.2)

    def pause(self):
        self.playing = False

    def stop(self):
        self.playing = False
        self.slider.set(0)
        self.render(0)

    def set_start(self):
        self.t_start = sec2hms(self.cur_time())
        self.update_range()

    def set_end(self):
        self.t_end = sec2hms(self.cur_time())
        self.update_range()

    def update_range(self):
        self.lbl_range.config(
            text=f'起始: {self.t_start or "--"}  结束: {self.t_end or "--"}')

    def confirm(self):
        self.playing = False
        self.on_confirm(self.t_start, self.t_end)
        self.cap.release()
        self.destroy()


# =====================================================================
# 主界面
# =====================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('视频转乐谱工具 v2.0')
        self.proc = None
        self.tmp_dir = None
        self.timer_on = False
        self.t0 = 0
        self.build_ui()

    # ---------------- UI 构建 ----------------
    def build_ui(self):
        pad = dict(padx=6, pady=3)
        main = ttk.Frame(self); main.pack(fill='both', expand=True, padx=8, pady=8)

        # 1. 文件设置
        f1 = ttk.LabelFrame(main, text='文件设置'); f1.pack(fill='x', **pad)
        self.var_video = tk.StringVar()
        self.var_output = tk.StringVar()
        ttk.Label(f1, text='视频文件:').grid(row=0, column=0, sticky='e')
        ttk.Entry(f1, textvariable=self.var_video, width=60).grid(row=0, column=1)
        ttk.Button(f1, text='浏览', command=self.pick_video).grid(row=0, column=2)
        ttk.Label(f1, text='输出文件:').grid(row=1, column=0, sticky='e')
        ttk.Entry(f1, textvariable=self.var_output, width=60).grid(row=1, column=1)
        ttk.Button(f1, text='浏览', command=self.pick_output).grid(row=1, column=2)

        # 2. 基本设置
        f2 = ttk.LabelFrame(main, text='基本设置'); f2.pack(fill='x', **pad)
        self.var_title = tk.StringVar()
        self.var_subtitle = tk.StringVar()
        self.var_notation = tk.StringVar(value='auto')
        ttk.Label(f2, text='标题:').grid(row=0, column=0, sticky='e')
        ttk.Entry(f2, textvariable=self.var_title, width=25).grid(row=0, column=1)
        ttk.Label(f2, text='副标题:').grid(row=0, column=2, sticky='e')
        ttk.Entry(f2, textvariable=self.var_subtitle, width=25).grid(row=0, column=3)
        nf = ttk.Frame(f2); nf.grid(row=1, column=0, columnspan=4, sticky='w')
        ttk.Label(nf, text='乐谱类型:').pack(side='left')
        for txt, val in [('自动检测', 'auto'), ('五线谱', 'staff'),
                         ('六线谱(TAB)', 'tab'), ('五六线谱对照', 'both')]:
            ttk.Radiobutton(nf, text=txt, value=val,
                            variable=self.var_notation,
                            command=self.update_output_name).pack(side='left', padx=4)

        # 3. 裁剪设置
        f3 = ttk.LabelFrame(main, text='裁剪设置'); f3.pack(fill='x', **pad)
        self.var_crop_mode = tk.StringVar(value='auto')
        self.var_y0 = tk.StringVar(); self.var_y1 = tk.StringVar()
        self.var_r0 = tk.StringVar(value='0.55'); self.var_r1 = tk.StringVar(value='1.0')
        ttk.Radiobutton(f3, text='自动检测', value='auto',
                        variable=self.var_crop_mode).grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(f3, text='像素坐标', value='pixel',
                        variable=self.var_crop_mode).grid(row=1, column=0, sticky='w')
        ttk.Label(f3, text='Y起始:').grid(row=1, column=1)
        ttk.Entry(f3, textvariable=self.var_y0, width=8).grid(row=1, column=2)
        ttk.Label(f3, text='Y结束:').grid(row=1, column=3)
        ttk.Entry(f3, textvariable=self.var_y1, width=8).grid(row=1, column=4)
        ttk.Button(f3, text='框选', command=self.open_crop_selector).grid(row=1, column=5)
        ttk.Radiobutton(f3, text='比例裁剪', value='ratio',
                        variable=self.var_crop_mode).grid(row=2, column=0, sticky='w')
        ttk.Label(f3, text='起始:').grid(row=2, column=1)
        ttk.Entry(f3, textvariable=self.var_r0, width=8).grid(row=2, column=2)
        ttk.Label(f3, text='结束:').grid(row=2, column=3)
        ttk.Entry(f3, textvariable=self.var_r1, width=8).grid(row=2, column=4)

        # 4. 时间范围
        f4 = ttk.LabelFrame(main, text='时间范围(可选)'); f4.pack(fill='x', **pad)
        self.var_start = tk.StringVar(); self.var_end = tk.StringVar()
        ttk.Label(f4, text='起始:').pack(side='left')
        ttk.Entry(f4, textvariable=self.var_start, width=10).pack(side='left')
        ttk.Label(f4, text='结束:').pack(side='left', padx=(10, 0))
        ttk.Entry(f4, textvariable=self.var_end, width=10).pack(side='left')
        ttk.Button(f4, text='选择时间', command=self.open_time_selector).pack(side='left', padx=10)

        # 5+6+7. 参数区（三列并排）
        prm = ttk.Frame(main); prm.pack(fill='x', **pad)
        f5 = ttk.LabelFrame(prm, text='高级设置'); f5.pack(side='left', fill='y', padx=2)
        self.var_interval = tk.StringVar(value='0.5')
        self.var_dpi = tk.StringVar(value='600')
        self.var_group_t = tk.StringVar(value='0.95')
        self.var_merge_t = tk.StringVar(value='0.95')
        self.var_dedup_t = tk.StringVar(value='0.65')
        for i, (lbl, var) in enumerate([('截取间隔(秒)', self.var_interval),
                                        ('输出DPI', self.var_dpi),
                                        ('分组阈值', self.var_group_t),
                                        ('合并阈值', self.var_merge_t),
                                        ('去重阈值', self.var_dedup_t)]):
            ttk.Label(f5, text=lbl + ':').grid(row=i, column=0, sticky='e')
            ttk.Entry(f5, textvariable=var, width=8).grid(row=i, column=1)

        f6 = ttk.LabelFrame(prm, text='图像增强'); f6.pack(side='left', fill='y', padx=2)
        self.var_p_lo = tk.StringVar(value='0.5')
        self.var_p_hi = tk.StringVar(value='99.5')
        self.var_sharpen = tk.StringVar(value='1.2')
        self.var_sigma = tk.StringVar(value='1.0')
        for i, (lbl, var) in enumerate([('拉伸低(%)', self.var_p_lo),
                                        ('拉伸高(%)', self.var_p_hi),
                                        ('锐化强度', self.var_sharpen),
                                        ('锐化核', self.var_sigma)]):
            ttk.Label(f6, text=lbl + ':').grid(row=i, column=0, sticky='e')
            ttk.Entry(f6, textvariable=var, width=8).grid(row=i, column=1)

        f7 = ttk.LabelFrame(prm, text='页边距(mm)'); f7.pack(side='left', fill='y', padx=2)
        self.var_mt = tk.StringVar(value='12'); self.var_mb = tk.StringVar(value='12')
        self.var_ml = tk.StringVar(value='10'); self.var_mr = tk.StringVar(value='10')
        for i, (lbl, var) in enumerate([('上', self.var_mt), ('下', self.var_mb),
                                        ('左', self.var_ml), ('右', self.var_mr)]):
            ttk.Label(f7, text=lbl + ':').grid(row=i, column=0, sticky='e')
            ttk.Entry(f7, textvariable=var, width=8).grid(row=i, column=1)

        # 8. 按钮区（含删除临时文件复选框 — 来自1.0版）
        f8 = ttk.Frame(main); f8.pack(fill='x', **pad)
        self.btn_start = ttk.Button(f8, text='开始处理', command=self.start)
        self.btn_start.pack(side='left')
        self.btn_stop = ttk.Button(f8, text='停止', command=self.stop, state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        ttk.Button(f8, text='打开输出目录', command=self.open_out_dir).pack(side='left', padx=4)
        ttk.Button(f8, text='打开临时文件夹', command=self.open_tmp_dir).pack(side='left', padx=4)
        ttk.Button(f8, text='保存配置', command=self.save_config).pack(side='left', padx=4)
        ttk.Button(f8, text='加载配置', command=self.load_config).pack(side='left')

        self.var_delete_temp = tk.BooleanVar(value=False)
        ttk.Checkbutton(f8, text='处理完成后删除临时文件夹',
                        variable=self.var_delete_temp).pack(side='right', padx=6)

        # 9. 状态区
        f9 = ttk.LabelFrame(main, text='处理状态'); f9.pack(fill='both', expand=True, **pad)
        st = ttk.Frame(f9); st.pack(fill='x')
        self.lbl_status = ttk.Label(st, text='就绪'); self.lbl_status.pack(side='left')
        self.lbl_step = tk.Label(st, text='', fg='blue'); self.lbl_step.pack(side='left', padx=10)
        self.lbl_timer = tk.Label(st, text='00:00:00', fg='green',
                                  font=('Consolas', 11)); self.lbl_timer.pack(side='right')
        self.progress = ttk.Progressbar(f9, mode='indeterminate')
        self.progress.pack(fill='x', padx=4, pady=2)
        self.log = ScrolledText(f9, height=12, state='disabled',
                                font=('Consolas', 9))
        self.log.pack(fill='both', expand=True, padx=4, pady=4)

    # ---------------- 文件选择 ----------------
    def pick_video(self):
        fp = filedialog.askopenfilename(
            filetypes=[('视频文件', '*.mp4 *.avi *.mkv *.mov *.flv'), ('所有文件', '*')])
        if fp:
            self.var_video.set(fp)
            self.var_title.set(os.path.splitext(os.path.basename(fp))[0])
            self.update_output_name()

    def update_output_name(self):
        v = self.var_video.get()
        if not v:
            return
        n = self.var_notation.get()
        suffix = '' if n == 'auto' else '_' + n
        self.var_output.set(os.path.splitext(v)[0] + suffix + '_乐谱.pdf')

    def pick_output(self):
        fp = filedialog.asksaveasfilename(defaultextension='.pdf',
                                          filetypes=[('PDF', '*.pdf')])
        if fp:
            self.var_output.set(fp)

    def open_crop_selector(self):
        if not self.var_video.get():
            messagebox.showwarning('提示', '请先选择视频文件')
            return
        def cb(y0, y1):
            self.var_y0.set(str(y0)); self.var_y1.set(str(y1))
            self.var_crop_mode.set('pixel')
        CropSelectorWindow(self, self.var_video.get(), cb)

    def open_time_selector(self):
        if not self.var_video.get():
            messagebox.showwarning('提示', '请先选择视频文件')
            return
        def cb(t0, t1):
            if t0: self.var_start.set(t0)
            if t1: self.var_end.set(t1)
        TimeSelectorWindow(self, self.var_video.get(), cb)

    def open_out_dir(self):
        d = os.path.dirname(self.var_output.get()) or '.'
        if not os.path.isdir(d):
            messagebox.showwarning('提示', f'目录不存在: {d}')
            return
        self._open_dir(d)

    def open_tmp_dir(self):
        d = self.tmp_dir
        if not d:
            out = self.var_output.get()
            if out:
                guess = os.path.splitext(out)[0] + '_temp'
                if os.path.isdir(guess):
                    d = guess
        if not d or not os.path.isdir(d):
            messagebox.showinfo('提示', '未找到临时文件夹(可能尚未生成或已被删除)')
            return
        self._open_dir(d)

    def _open_dir(self, d):
        if sys.platform == 'win32':
            os.startfile(d)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', d])
        else:
            subprocess.Popen(['xdg-open', d])

    # ---------------- 配置保存/加载 ----------------
    def all_vars(self):
        result = {}
        for k, v in vars(self).items():
            if k.startswith('var_') and isinstance(v, (tk.StringVar, tk.BooleanVar)):
                result[k] = v
        return result

    def save_config(self):
        data = {}
        for k, v in self.all_vars().items():
            data[k] = v.get()
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log_line(f'配置已保存: {CONFIG_PATH}')

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            messagebox.showinfo('提示', '未找到配置文件')
            return
        with open(CONFIG_PATH, encoding='utf-8') as f:
            data = json.load(f)
        for k, v in self.all_vars().items():
            if k in data:
                try:
                    v.set(data[k])
                except tk.TclError:
                    pass
        self.log_line('配置已加载')

    # ---------------- 命令构建 ----------------
    def build_command(self):
        cmd = [sys.executable, EXTRACT_PY, self.var_video.get(),
               '-o', self.var_output.get()]
        if self.var_title.get():
            cmd += ['-t', self.var_title.get()]
        if self.var_subtitle.get():
            cmd += ['-s', self.var_subtitle.get()]
        cmd += ['--notation', self.var_notation.get()]
        # 间隔秒 → fps
        try:
            fps = 1.0 / float(self.var_interval.get())
        except (ValueError, ZeroDivisionError):
            fps = 2.0
        cmd += ['--fps', f'{fps:g}']
        mode = self.var_crop_mode.get()
        if mode == 'auto':
            cmd += ['--auto-crop']
        elif mode == 'pixel' and self.var_y0.get() and self.var_y1.get():
            cmd += ['--crop-y', f'{self.var_y0.get()}:{self.var_y1.get()}']
        elif mode == 'ratio':
            cmd += ['--crop', f'{self.var_r0.get()}:{self.var_r1.get()}']
        if self.var_start.get():
            cmd += ['--start', self.var_start.get()]
        if self.var_end.get():
            cmd += ['--end', self.var_end.get()]
        cmd += ['--group-t', self.var_group_t.get(),
                '--merge-t', self.var_merge_t.get(),
                '--dedup-t', self.var_dedup_t.get(),
                '--dpi', self.var_dpi.get(),
                '--margins', f'{self.var_mt.get()}:{self.var_mb.get()}:'
                             f'{self.var_ml.get()}:{self.var_mr.get()}',
                '--percentile', f'{self.var_p_lo.get()}:{self.var_p_hi.get()}',
                '--sharpen', self.var_sharpen.get(),
                '--sharpen-sigma', self.var_sigma.get()]
        if self.var_delete_temp.get():
            cmd += ['--delete-temp']
        else:
            cmd += ['--keep-temp']
        return cmd

    # ---------------- 处理控制 ----------------
    def start(self):
        if not self.var_video.get():
            messagebox.showwarning('提示', '请先选择视频文件')
            return
        if not self.var_output.get():
            messagebox.showwarning('提示', '请先设置输出文件路径')
            return
        if not os.path.exists(EXTRACT_PY):
            messagebox.showerror('错误', f'未找到 {EXTRACT_PY}')
            return
        self.tmp_dir = None
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.lbl_status.config(text='处理中...')
        self.lbl_step.config(text='')
        self.progress.start(15)
        self.t0 = time.time()
        self.timer_on = True
        self.tick()
        self.log_clear()
        cmd = self.build_command()
        self.log_line('> ' + ' '.join(cmd))
        threading.Thread(target=self.run_process, args=(cmd,), daemon=True).start()

    def stop(self):
        """停止处理 — 来自1.1版的进程终止方案（杀子进程树）"""
        if not (self.proc and self.proc.poll() is None):
            return
        pid = self.proc.pid
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass
        finally:
            try:
                self.proc.kill()
            except Exception:
                pass

        self.timer_on = False
        self.progress.stop()
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.lbl_status.config(text='已停止')
        self.lbl_step.config(text='')
        self.log_line('已终止处理进程（含全部子进程，如 ffmpeg）')

    def tick(self):
        if self.timer_on:
            self.lbl_timer.config(text=sec2hms(time.time() - self.t0))
            self.after(1000, self.tick)

    def run_process(self, cmd):
        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1
            )
            if sys.platform == 'win32':
                popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs['preexec_fn'] = os.setsid

            self.proc = subprocess.Popen(cmd, **popen_kwargs)

            for raw in iter(self.proc.stdout.readline, b''):
                try:
                    line = raw.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        line = raw.decode('gbk')
                    except UnicodeDecodeError:
                        line = raw.decode('utf-8', errors='replace')
                line = line.rstrip()

                if '临时文件保留在:' in line:
                    self.tmp_dir = line.split('临时文件保留在:')[-1].strip()
                elif '临时文件夹已删除:' in line:
                    self.tmp_dir = None

                m = re.search(r'\[(\d+)/(\d+)\]', line)
                if m:
                    self.after(0, self.lbl_step.config,
                               {'text': f'步骤 {m.group(1)}/{m.group(2)}'})
                self.after(0, self.log_line, line)

            self.proc.wait()
            ok = self.proc.returncode == 0
            self.after(0, self.on_complete, ok,
                       '处理完成' if ok else f'退出码 {self.proc.returncode}')
        except Exception as e:
            self.after(0, self.on_complete, False, str(e))

    def on_complete(self, success, message):
        self.timer_on = False
        self.progress.stop()
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.lbl_status.config(text='完成' if success else '失败')
        elapsed = sec2hms(time.time() - self.t0)
        self.log_line(f'--- {message}，耗时 {elapsed} ---')
        if success and self.tmp_dir:
            self.log_line(f'临时文件夹: {self.tmp_dir}')

    # ---------------- 日志 ----------------
    def log_line(self, text):
        self.log.config(state='normal')
        self.log.insert('end', text + '\n')
        self.log.see('end')
        self.log.config(state='disabled')

    def log_clear(self):
        self.log.config(state='normal')
        self.log.delete('1.0', 'end')
        self.log.config(state='disabled')


if __name__ == '__main__':
    App().mainloop()
