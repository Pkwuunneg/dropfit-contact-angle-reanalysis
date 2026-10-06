"""Optional standard-library Tk desktop launcher; computation runs in a worker thread."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog,messagebox,scrolledtext
import webbrowser

ROOT=Path(__file__).resolve().parent


def main():
    root=tk.Tk();root.title("PDF 接触角重分析 — DropFit");root.geometry("880x610")
    input_var=tk.StringVar(value=str(ROOT/"input_pdfs"))
    output_var=tk.StringVar(value=str(ROOT/"my_results"))
    config_var=tk.StringVar(value="")
    def row(label,var,select):
        frame=tk.Frame(root);frame.pack(fill="x",padx=14,pady=6)
        tk.Label(frame,text=label,width=11,anchor="w").pack(side="left")
        tk.Entry(frame,textvariable=var).pack(side="left",fill="x",expand=True)
        tk.Button(frame,text="选择",command=select).pack(side="left",padx=7)
    def pick_dir(var):
        path=filedialog.askdirectory()
        if path:var.set(path)
    row("输入文件夹",input_var,lambda:pick_dir(input_var))
    row("输出文件夹",output_var,lambda:pick_dir(output_var))
    def pick_config():
        path=filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if path:config_var.set(path)
    row("复核配置(可空)",config_var,pick_config)
    tk.Label(root,text="默认沿用报告绿色基准线。结果是条件性重拟合，不是新的独立测量。",anchor="w").pack(fill="x",padx=14,pady=5)
    log=scrolledtext.ScrolledText(root,height=22);log.pack(fill="both",expand=True,padx=14,pady=8)
    events=queue.Queue()
    def launch():
        button.config(state="disabled");log.delete("1.0","end")
        source,destination,config=input_var.get(),output_var.get(),config_var.get() or None
        def work():
            try:
                from dropfit.__main__ import batch
                batch(source,destination,config,progress=lambda s:events.put(("log",s)))
                events.put(("done",str(Path(destination).resolve()/"report.html")))
            except Exception as exc: events.put(("error",str(exc)))
        threading.Thread(target=work,daemon=True).start()
    button=tk.Button(root,text="开始分析（含敏感性检查）",command=launch);button.pack(pady=8)
    def poll():
        try:
            while True:
                kind,msg=events.get_nowait()
                if kind=="log":log.insert("end",msg+"\n");log.see("end")
                elif kind=="done":button.config(state="normal");webbrowser.open(Path(msg).as_uri())
                else:button.config(state="normal");messagebox.showerror("分析失败",msg)
        except queue.Empty:pass
        root.after(100,poll)
    poll();root.mainloop()

if __name__=="__main__":main()
