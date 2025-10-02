import datetime
from typing import Dict, List


def generate_disagreement_report(
    query: str, results: Dict[str, List[str]], metrics: Dict[str, float], output_path: str
):
    """
    Generates an HTML report for the disagreement analysis.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
    <html>
    <head>
        <title>Disagreement Report for "{query}"</title>
        <style>
            body {{ font-family: sans-serif; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Disagreement Report</h1>
        <p><strong>Query:</strong> {query}</p>
        <p><strong>Generated on:</strong> {now}</p>

        <h2>Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """

    for metric_name, value in metrics.items():
        html_content += f"<tr><td>{metric_name}</td><td>{value:.4f}</td></tr>"

    html_content += """
        </table>

        <h2>Retrieval Results</h2>
    """

    for retriever_name, result_list in results.items():
        html_content += f"<h3>{retriever_name}</h3>"
        html_content += "<table><tr><th>Rank</th><th>Document</th></tr>"
        for i, doc in enumerate(result_list):
            html_content += f"<tr><td>{i+1}</td><td>{doc}</td></tr>"
        html_content += "</table>"

    html_content += """
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)
