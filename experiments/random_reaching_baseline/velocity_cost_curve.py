#!/usr/bin/env python3
"""One CPU-only velocity-cost figure; --plot-only regenerates from compact CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = [("M0",2.25,.1,.1),("M1",2.25,1.,.1),
         ("M4",2.,.2,.05),("M5",2.,.6,.05)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE)
    parser.add_argument("--plot-only",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    csv_path = args.output/"velocity_cost_curve.csv"
    if not args.plot_only:
        from cost_scaling import barrier,HELPER_SHA256
        rows = []
        for name,bound,weight,qd in CASES:
            center = barrier(0.,-bound,bound)[0]
            for step in range(271):
                speed = step/100
                limit = weight*(barrier(speed,-bound,bound)[0]-center)
                ordinary = .5*qd*speed*speed
                rows.append(dict(configuration=name,velocity_rad_s=speed,
                    cost_bound_rad_s=bound,vel_lim_cost=weight,qd_cost=qd,
                    relative_limit_cost=limit,ordinary_qd_cost=ordinary,
                    total_velocity_cost=limit+ordinary))
        with csv_path.open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(rows)
        metadata = dict(scope="One joint at one knot: reviewed C2 limit cost relative to center plus ordinary qd quadratic",
            helper_sha256=HELPER_SHA256,csv_sha256=sha(csv_path),script_sha256=sha(__file__),
            curves=[dict(configuration=n,cost_bound_rad_s=b,vel_lim_cost=w,qd_cost=q) for n,b,w,q in CASES],
            speed_grid_rad_s=dict(start=0.,stop=2.7,step=.01),physical_limit_rad_s=2.5,
            plotting_only="No objective ranking across methods; each curve uses that method's scalar weights. Other objective terms and dynamics excluded.")
        (args.output/"velocity_cost_curve.json").write_text(json.dumps(metadata,indent=2)+"\n")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    colors = {"M0":"#2166ac","M1":"#b35806","M4":"#1b7837","M5":"#762a83"}
    styles = {"M0":"-","M1":"--","M4":"-","M5":"--"}
    fig,ax = plt.subplots(figsize=(9.0,5.0))
    inset = ax.inset_axes([.12,.43,.43,.47])
    for name,bound,weight,qd in CASES:
        selected = [r for r in rows if r["configuration"]==name]
        x = [float(r["velocity_rad_s"]) for r in selected]
        y = [float(r["total_velocity_cost"]) for r in selected]
        label = f"{name}: limit {weight:g}, qd {qd:g}"
        for target in (ax,inset):
            target.plot(x,y,color=colors[name],linestyle=styles[name],linewidth=1.8,
                        label=label if target is ax else None)
    for bound,label,style in [(2.,"M4/M5 cost bound",":"),(2.25,"M0/M1 cost bound","--")]:
        ax.axvline(bound,color="0.5",linestyle=style,linewidth=1)
        ax.text(bound-.015,.97,label,transform=ax.get_xaxis_transform(),rotation=90,
                va="top",ha="right",fontsize=8,color="0.35")
    ax.axvline(2.5,color="#b2182b",linestyle="-.",linewidth=1)
    ax.text(2.5+.015,.97,"Physical limit",transform=ax.get_xaxis_transform(),rotation=90,
            va="top",ha="left",fontsize=8,color="#b2182b")
    ax.set(xlim=(0.,2.7),ylim=(0.,None),xlabel="Joint speed [rad/s]",
           ylabel="Relative velocity cost per joint and knot")
    ax.grid(alpha=.2)
    fig.suptitle("Velocity penalties: ordinary-motion drag and boundary deterrence",fontsize=12,y=.98)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,.93),
               ncol=4,fontsize=8,frameon=False)
    inset.set(xlim=(0.,1.5),ylim=(0.,.75),title="Low-speed detail (same linear cost)")
    inset.title.set_fontsize(9);inset.tick_params(labelsize=8);inset.grid(alpha=.2)
    inset.xaxis.set_major_locator(MaxNLocator(4));inset.yaxis.set_major_locator(MaxNLocator(4))
    fig.text(.12,.025,"C2 continuation uses 1% of each cost interval. Curves omit pose/effort costs and dynamics.",fontsize=8)
    fig.subplots_adjust(left=.12,right=.97,bottom=.15,top=.83)
    fig.savefig(args.output/"velocity_cost_curve.png",dpi=180)
    fig.savefig(args.output/"velocity_cost_curve.pdf")
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output),rows=len(rows))))


if __name__ == "__main__":
    main()
