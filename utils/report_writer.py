# utils/report_writer.py

def write_report(metrics_dict, filename="report.txt"):
    with open(filename, "w") as f:
        for metric, value in metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")