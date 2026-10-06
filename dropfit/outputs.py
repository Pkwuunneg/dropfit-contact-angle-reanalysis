"""Audit figures and self-contained HTML / CSV / JSON / LaTeX exports."""
from __future__ import annotations
from pathlib import Path
import base64
import csv
import html
import json
from collections import Counter
import numpy as np
from PIL import Image

STATUS_TEXT={"CONDITIONAL":"条件有效：需确认基准线与模型", "REVIEW":"待复核：不自动采用", "NA":"无法可靠测角"}
FLAG_ZH={
 "SHARED_REPORT_BASELINE":"沿用报告的绿色基准线，并非完全独立测量",
 "BASELINE_NOT_INDEPENDENTLY_VALIDATED":"基准线尚未经独立人工确认",
 "UNRESOLVED_HEIGHT":"液滴原生像素高度不足12 px",
 "LOCAL_GLOBAL_DISAGREEMENT":"局部与全局拟合存在明显差异",
 "LOCAL_ASYMMETRY_OR_EDGE_ARTIFACT":"左右局部角差较大，可能为形态或标线误差",
 "LOCAL_DERIVATIVE_PARAMETER_SENSITIVE":"局部导数对窗口/阶数等参数敏感",
 "LOCAL_FIT_UNRESOLVED":"局部切线拟合不可靠",
 "CIRCLE_MODEL_RESIDUAL_HIGH":"全轮廓圆拟合残差过大",
 "CIRCLE_PARAMETER_SENSITIVE":"圆拟合角对处理参数敏感",
 "SENSITIVITY_RUN_FAILED":"部分敏感性测试未能拟合",
 "EXACT_DUPLICATE_RASTER":"与另一记录的原生RGB图像完全相同",
 "SEVERE_ANNOTATION_OCCLUSION":"大量轮廓被彩色标线遮挡",
 "DETACHED_OR_WRONG_COMPONENT":"所选暗色区域未接触基准线",
 "CLIPPED_DROP":"液滴触及选区边界，请重新框选",
 "BASELINE_OR_FIT_UNRESOLVED":"无法可靠确定基准线或拟合轮廓"}


def clean_json(value):
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray): return clean_json(value.tolist())
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,np.bool_): return bool(value)
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def number(x,decimals=2):
    return "NA" if x is None or not np.isfinite(x) else f"{x:.{decimals}f}"


def row_data(r):
    f=r.get("fit") or {};c=f.get("circle") or {};l=f.get("local") or {}
    s=(r.get("sensitivity") or {}).get("ranges",{}).get("circle_deg") or {}
    return {"source_file":r["filename"],"source_key":r["key"],"liquid":r["liquid"],
            "material":r["material"],"filename_method":r["filename_method"],
            "native_width_px":r["native_width"],"native_height_px":r["native_height"],
            "reported_mean_deg":r.get("reported_angles",{}).get("mean_deg"),
            "circle_candidate_deg":c.get("theta_deg"),"circle_rmse_px":c.get("rmse_px"),
            "local_left_candidate_deg":l.get("left",{}).get("theta_deg"),
            "local_right_candidate_deg":l.get("right",{}).get("theta_deg"),
            "local_mean_candidate_deg":l.get("mean_deg"),"theta_half_check_deg":f.get("theta_half_deg"),
            "circle_sensitivity_min_deg":s.get("min"),"circle_sensitivity_max_deg":s.get("max"),
            "circle_sensitivity_span_deg":s.get("span"),
            "observed_height_px":f.get("height_observed_px",r.get("image_diagnostics",{}).get("height_observed_px")),
            "masked_fraction":f.get("masked_fraction"),"status":r["status"],
            "reportable_conditional_theta_deg":r.get("reportable_theta_deg"),
            "baseline_source":(r.get("baseline") or {}).get("source"),
            "baseline_validated":(r.get("baseline") or {}).get("validated",False),
            "difference_from_report_deg":r.get("difference_from_report_deg"),
            "exact_duplicate_of":r.get("exact_duplicate_of",""),
            "flags":";".join(r["flags"]),"info_flags":";".join(r["info_flags"]),
            "image_sha256":r["image_sha256"]}


def save_diagnostics(record,result,diagnostics,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .geometry import Baseline
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(record.rgb).save(folder/"source.png")
    fig,ax=plt.subplots(figsize=(9.4,6.4),layout="constrained")
    ax.imshow(record.rgb)
    fit=result.get("fit")
    baseline=diagnostics["baseline"] if diagnostics else (Baseline(**result["baseline"]) if result.get("baseline") else None)
    if diagnostics:
        contour=diagnostics["contour"];p=contour["points_xy"];uv=contour["points_uv"];used=contour["used"]
        ax.plot(p[used,0],p[used,1],".",markersize=2,label="Source edges used")
        if (~used).any(): ax.plot(p[~used,0],p[~used,1],"x",markersize=2.5,alpha=.6,label="Excluded colour edges")
        c=fit["circle"];theta=np.radians(c["theta_deg"]);phi=np.linspace(-theta,theta,500)
        curve=np.column_stack((c["u_center_px"]+c["radius_px"]*np.sin(phi),
                               c["v_center_px"]+c["radius_px"]*np.cos(phi)))
        xy=baseline.xy(curve);ax.plot(xy[:,0],xy[:,1],linewidth=1.6,label="Robust full-contour circle")
        if fit["local"]:
            for side in ("left","right"):
                model=fit["local"][side];q=np.linspace(model["q_contact"],model["q_max_observed"],120)
                uvline=np.column_stack((np.polynomial.polynomial.polyval(q,model["coeff_u"]),
                                       np.polynomial.polynomial.polyval(q,model["coeff_v"])))
                xy=baseline.xy(uvline);ax.plot(xy[:,0],xy[:,1],"--",linewidth=1.7,label=f"{side.title()} local polynomial")
        span=max(20.,np.ptp(p[:,0])*.12)
        ax.set_xlim(max(0,p[:,0].min()-span),min(record.rgb.shape[1],p[:,0].max()+span))
        ax.set_ylim(min(record.rgb.shape[0],p[:,1].max()+span),max(0,p[:,1].min()-span))
        residual=np.hypot(uv[:,0]-c["u_center_px"],uv[:,1]-c["v_center_px"])-c["radius_px"]
        data=np.column_stack((p,uv,used.astype(int),residual))
        np.savetxt(folder/"contour.csv",data,delimiter=",",header="x_px,y_px,u_px,v_px,used,circle_signed_residual_px",comments="",fmt="%.6f")
        residual_fig,resax=plt.subplots(figsize=(8,3.7),layout="constrained")
        resax.plot(uv[used,0],residual[used],".",markersize=3,label="Observed residual")
        resax.axhline(0,linestyle="--",linewidth=.8)
        resax.set_xlabel("u along substrate (native px)");resax.set_ylabel("Radial residual (px)")
        resax.set_title("Signed residuals: systematic structure matters, not just RMSE")
        resax.grid(True,alpha=.2);residual_fig.savefig(folder/"residual.png",dpi=140);plt.close(residual_fig)
    elif baseline:
        y=baseline.origin[1];ax.set_ylim(min(record.rgb.shape[0],y+95),max(0,y-95))
    if baseline:
        xlim=ax.get_xlim();u=np.array([[-record.rgb.shape[1],0],[record.rgb.shape[1],0]])
        xy=baseline.xy(u);ax.plot(xy[:,0],xy[:,1],"-.",linewidth=1,label="Baseline used (not independently inferred)")
        ax.set_xlim(xlim)
    title=f"{result['status']} | "
    if fit:
        title+=f"circle {fit['circle']['theta_deg']:.2f} deg; RMSE {fit['circle']['rmse_px']:.2f} px"
    else: title+="No reportable finite angle"
    ax.set_title(title);ax.set_xlabel("Native image x (px)");ax.set_ylabel("Native image y (px)")
    handles,labels=ax.get_legend_handles_labels()
    if handles: ax.legend(loc="upper right",fontsize=7,framealpha=.95)
    fig.savefig(folder/"diagnostic.png",dpi=150,bbox_inches="tight",pad_inches=.12);plt.close(fig)
    write_json(folder/"result.json",result)


def _img_tag(path,alt):
    data=base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def tex_escape(s):
    substitutions={"\\":r"\textbackslash{}","_":r"\_","%":r"\%","&":r"\&","#":r"\#","{":r"\{","}":r"\}","$":r"\$","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return "".join(substitutions.get(c,c) for c in str(s))


def write_outputs(results,output,manifest):
    output=Path(output);rows=[row_data(r) for r in results]
    write_json(output/"results.json",results);write_json(output/"manifest.json",manifest)
    if rows:
        with (output/"results.csv").open("w",encoding="utf-8-sig",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:
                writer.writerow({k:("NA" if v is None else f"{v:.6f}" if isinstance(v,(float,np.floating)) else v) for k,v in row.items()})
    counts=Counter(r["status"] for r in results);duplicates=sum(bool(r.get("exact_duplicate_of")) for r in results)
    findings=["# 自动审计结论", "", "以下结论仅针对当前图片和给定基准线，不是对材料本征性质的自动判定。", ""]
    finite=[r for r in results if r.get("fit")]
    if finite:
        rmse=[r["fit"]["circle"]["rmse_px"] for r in finite]
        findings.append(f"在 {len(finite)} 个能够产生拟合候选值的记录中，圆模型残差RMSE为 {min(rmse):.2f}--{max(rmse):.2f} px。小残差只能说明所提取轮廓与圆模型接近，不能证明基准线或被标线遮挡的真实轮廓正确。")
    for r in results:
        if r["status"] in ("REVIEW","NA"):
            explanation="；".join(FLAG_ZH.get(flag,flag) for flag in r["flags"])
            findings.extend(["",f"{r['filename']}：{r['status']}。{explanation}。"])
        if r.get("exact_duplicate_of"):
            findings.extend(["",f"{r['filename']} 的原生RGB像素与 {r['exact_duplicate_of']} 完全相同，不应作为新增独立样本。"])
    groups={}
    for r in finite:
        groups.setdefault((r["liquid"],r["material"]),[]).append(r)
    for (liquid,material),group in groups.items():
        values=[r["fit"]["circle"]["theta_deg"] for r in group]
        if len(values)>1 and max(values)-min(values)>10:
            findings.extend(["",f"{liquid}--{material} 的不同图像记录给出 {min(values):.2f}--{max(values):.2f}° 的圆拟合候选范围。必须核查编号、取样位置、液滴和读数时刻；仅凭角差不能删除其中一项，也不能把它们直接平均为同一条件的重复测量。"])
    findings.extend(["", "REVIEW可由标线压缩伪影、模型不适合或真实不对称等不同原因触发，本程序不能仅凭这些图像唯一识别原因。没有自动推断完全润湿、蒸发、渗透或材料表面能。"])
    (output/"conclusions.md").write_text("\n".join(findings),encoding="utf-8")
    summary=["# PDF接触角图像重分析结果", "",
             f"共处理 {len(results)} 个图像记录：CONDITIONAL {counts['CONDITIONAL']}，REVIEW {counts['REVIEW']}，NA {counts['NA']}。精确重复栅格 {duplicates} 个。", "",
             "这是对同一组图片的模型/算法复核，不是新增独立实验。以文件名确定液体、材料与原算法标签；PDF内的材料、液体默认字段不用于分类。",
             "CONDITIONAL 表示在给定基准线与圆形轮廓假设下通过程序的筛查；不是已确认的真实平衡角。REVIEW 的拟合值仅供诊断，不自动采用。NA 不是0度。", "",
             "敏感性上下限来自基准线、灰度阈值、排除带、局部窗口与阶数的一次一因素扰动，不是95%置信区间，也不是独立重复测量的标准差。", "",
             "|文件|原报告角/度|圆拟合候选/度|局部候选均值/度|圆拟合敏感性范围/度|状态|",
             "|---|---:|---:|---:|---|---|"]
    table_html=[];details=[]
    texrows=[]
    for r,row in zip(results,rows):
        sensitivity=f"{number(row['circle_sensitivity_min_deg'])}--{number(row['circle_sensitivity_max_deg'])}"
        vals=[r["filename"],number(row["reported_mean_deg"]),number(row["circle_candidate_deg"]),
              number(row["local_mean_candidate_deg"]),sensitivity,r["status"]]
        summary.append("|"+"|".join(vals)+"|")
        table_html.append("<tr>"+"".join(f"<td>{html.escape(x)}</td>" for x in vals)+"</tr>")
        flags=r["flags"]+r["info_flags"]
        flagtext="；".join(FLAG_ZH.get(flag,flag) for flag in flags)
        folder=output/r["output_folder"]
        diagnostic=_img_tag(folder/"diagnostic.png","拟合轮廓与原生图像叠加")
        residual=_img_tag(folder/"residual.png","圆拟合残差") if (folder/"residual.png").exists() else ""
        duplicate=f"<p>精确重复于：{html.escape(r['exact_duplicate_of'])}</p>" if r.get("exact_duplicate_of") else ""
        details.append(f"<details><summary>{html.escape(r['filename'])} — {r['status']}</summary>"
                       f"<p>{html.escape(flagtext)}</p>{duplicate}<p>{html.escape(r.get('error',''))}</p>{diagnostic}{residual}</details>")
        texrows.append(" & ".join([tex_escape(r["filename"].removesuffix(".pdf")),number(row["reported_mean_deg"],1),
                                  number(row["circle_candidate_deg"],1),number(row["local_mean_candidate_deg"],1),r["status"]])+r" \\")
    summary.extend(["", "## 解释边界", "",
        "不能用拟合与原软件读数的一致性，替代基准线、轮廓和模型的正确性检验。不能仅因某个样品与其他液滴角度差大而自动删除。",
        "对于同一材料下不同编号差异较大的结果，须先核对是否为不同液滴、位置、读数时刻或样品状态；程序不以原始角差决定NA。",
        "精确重复只在原生RGB数组及尺寸完全一致时标记；不同标线的同一液滴不会被误称为独立重复测量。未将文件条目数作为统计样本量。",
        "未从单张图推断完全润湿、蒸发速率、渗透速率、表面能或表面张力。原始长度/面积单位未经标定，因此所有长度仅使用原生像素。"])
    (output/"summary.md").write_text("\n".join(summary),encoding="utf-8")
    explanation="""<p>本页由确定性规则生成。拟合程序没有读入原软件的接触角作为初值、目标或筛选条件。默认主结果是<strong>全轮廓鲁棒圆拟合</strong>，局部参数多项式提供左右切线复核。</p>
<p>图片中的彩色线已经栅格化。程序不对被遮挡的真实轮廓做AI补全，也不声称恢复出原始相机数据。自动基准线来自报告的绿色标线，必须人工检查是否与实际固体表面重合。</p>
<p><strong>CONDITIONAL</strong>：在给定基准线/球冠近似下通过筛查；<strong>REVIEW</strong>：保留候选值但不自动采用；<strong>NA</strong>：无可靠有限角，不表示0°。误差门限是可配置的工程筛查规则，不是国家标准。</p>
<p>圆拟合范围是处理参数扰动得到的敏感性范围，不是置信区间。局部导数可能比全局拟合敏感。相同图片的重拟合并不增加独立样本数。</p>"""
    document="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>接触角图像重分析</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;line-height:1.75;color:#1f2933}h1,h2{line-height:1.35}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 10px;border-bottom:1px solid #ddd;text-align:left}th{background:#f1f5f9}details{margin:16px 0;border:1px solid #ddd;padding:12px}summary{font-weight:600;cursor:pointer}img{display:block;max-width:100%;height:auto;margin:12px auto}code{background:#f1f5f9;padding:2px 4px}.scroll{overflow:auto}</style>
<h1>PDF接触角图像重分析</h1>"""
    document+=f"<p>图像记录 {len(results)}；条件有效 {counts['CONDITIONAL']}；待复核 {counts['REVIEW']}；NA {counts['NA']}；精确重复 {duplicates}。</p>"+explanation
    document+='<h2>1. 数值汇总</h2><div class="scroll"><table><thead><tr>'+"".join(f"<th>{x}</th>" for x in ["源文件","原报告角/°","圆拟合候选/°","局部均值候选/°","圆拟合敏感性范围/°","状态"])+"</tr></thead><tbody>"+"".join(table_html)+"</tbody></table></div>"
    document+="<h2>2. 逐图审计</h2><p>点击每个文件展开。原图中的彩色辅助线仍留在底图上；新增拟合线以图例与线型区分。残差横坐标为沿基准线的像素坐标。</p>"+"".join(details)
    document+="<h2>3. 自动审计结论</h2>"+"".join("<p>"+html.escape(line)+"</p>" for line in findings[2:] if line)
    document+="<h2>4. 使用限制</h2><p>本次输出只支持对图像几何、拟合一致性与可测性的讨论。它不能单独验证样品身份、读数时刻或真实润湿机制。提取出的原生图像、全部轮廓点、逐次扰动结果和环境信息均保存在配套目录。</p></html>"
    (output/"report.html").write_text(document,encoding="utf-8")
    tex=r"""% Requires ctex (or another Chinese-capable setup), booktabs, longtable.
% All numbers are algorithmic reanalyses, not new independent measurements.
\subsection{PDF液滴图像的数字轮廓重分析}
为复核软件读数，从实验PDF报告中提取原生栅格图像，并在给定固--液基准线下提取液滴上轮廓。对可辨认的轮廓点进行全轮廓鲁棒圆拟合，同时采用局部参数多项式拟合估计左右表观接触角。原报告数值仅用于事后比较，不参与拟合或异常判定。

自动模式使用报告中已标注的绿色基准线，因此属于共享基准线的算法交叉验证，而不是完全独立的测角。图中的彩色辅助线被作为遮挡筛除，但JPEG压缩和边缘覆盖造成的信息损失无法完全消除。

\begingroup\small
\setlength{\tabcolsep}{4pt}
\begin{longtable}{p{0.38\linewidth}rrrl}
\caption{原读数与图像重分析候选值比较。角度单位为度；REVIEW行不自动纳入有效测量结果。}\\
\toprule 源文件标签 & 原读数 & 圆拟合 & 局部均值 & 状态\\\midrule
\endfirsthead
\toprule 源文件标签 & 原读数 & 圆拟合 & 局部均值 & 状态\\\midrule
\endhead
"""+"\n".join(texrows)+r"""
\bottomrule\end{longtable}\endgroup

其中CONDITIONAL表示在既定基准线与圆形轮廓假设下通过程序筛查，仍需人工确认；REVIEW表示模型差异、局部不对称或参数敏感性需要复核；NA表示未获得可靠有限角，而非接触角严格为零。基准线移动、灰度阈值改变及拟合窗口调整所得到的范围仅表示处理参数敏感性，不是统计置信区间。

同一图像的不同算法结果不是独立重复测量。没有据此计算材料的总体均值、总体标准差或显著性检验，也未反演表面能、表面张力或未经标定的毫米尺度。
"""
    (output/"fit_summary.tex").write_text(tex,encoding="utf-8")
