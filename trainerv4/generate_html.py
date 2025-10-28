import sqlite3
import json
import argparse
import os
from datetime import date, datetime

# --- Configuration ---
DEFAULT_DB_FILENAME = "stock_analysis.db"
DEFAULT_HTML_FILENAME = "stock_analysis_report.html"

# --- Helper Functions ---

def get_most_recent_date(db_path):
    """Finds the most recent analysis_date in the database."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(analysis_date) FROM stock_analysis")
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
    except sqlite3.Error as e:
        print(f"Error finding most recent date in DB: {e}")
        return None

def fetch_analysis_data(db_path, analysis_date):
    """Fetches analysis data for a specific date, ordered by bullishness."""
    data = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row # Return rows as dict-like objects
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    ticker, model_confidence_win, bullishness_score, reasoning,
                    news_sentiment_score, growth_score, valuation_score,
                    financial_health_score, positive_summary, negative_summary,
                    headlines, last_updated
                FROM stock_analysis
                WHERE analysis_date = ?
                ORDER BY bullishness_score DESC, ticker ASC
            """, (analysis_date,))
            data = [dict(row) for row in cursor.fetchall()]
            print(f"Fetched {len(data)} records for date {analysis_date}.")
            return data
    except sqlite3.Error as e:
        print(f"Error fetching data from database for date {analysis_date}: {e}")
        return []

def generate_score_cell(score, max_score=10):
    """Generates an HTML table cell with color based on score."""
    if score is None:
        return '<td class="border px-4 py-2 text-center text-gray-500">-</td>'

    # Determine color based on score percentage
    percentage = (score / max_score) * 100
    color_class = "bg-gray-200 text-gray-800" # Default/Neutral (e.g., score 5)
    if percentage >= 70: # High scores (7-10) -> Green
        color_class = "bg-green-100 text-green-800"
    elif percentage >= 40: # Mid scores (4-6) -> Yellow/Orange
         color_class = "bg-yellow-100 text-yellow-800"
    else: # Low scores (1-3) -> Red
        color_class = "bg-red-100 text-red-800"

    return f'<td class="border px-4 py-2 text-center font-semibold {color_class}">{score}</td>'

def generate_html_report(data, analysis_date, output_filename):
    """Generates the static HTML report file."""
    if not data:
        print("No data provided to generate report.")
        return

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Report - {analysis_date}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        /* Add fixed header styles if needed */
        /* th {{ position: sticky; top: 0; background-color: #f7fafc; z-index: 10; }} */
    </style>
     <link rel="preconnect" href="https://rsms.me/">
     <link rel="stylesheet" href="https://rsms.me/inter/inter.css">
</head>
<body class="bg-gray-100 p-4 md:p-8">
    <div class="container mx-auto bg-white p-6 rounded-lg shadow-md">
        <h1 class="text-3xl font-bold mb-2 text-gray-800">Stock Analysis Report</h1>
        <p class="text-lg text-gray-600 mb-6">Analysis Date: <span class="font-semibold">{analysis_date}</span></p>

        <div class="overflow-x-auto">
            <table class="min-w-full bg-white border border-gray-300 table-auto">
                <thead class="bg-gray-100">
                    <tr>
                        <th class="px-4 py-2 border text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Ticker</th>
                        <th class="px-4 py-2 border text-center text-sm font-semibold text-gray-700 uppercase tracking-wider">Model Conf.</th>
                        <th class="px-4 py-2 border text-center text-sm font-semibold text-gray-700 uppercase tracking-wider">Overall Score</th>
                        <th class="px-4 py-2 border text-center text-sm font-semibold text-gray-700 uppercase tracking-wider">News Score</th>
                        <th class="px-4 py-2 border text-center text-sm font-semibold text-gray-700 uppercase tracking-wider">Growth Score</th>
                        <th class="px-4 py-2 border text-center text-sm font-semibold text-gray-700 uppercase tracking-wider">Valuation Score</th>
                        <th class="px-4 py-2 border text-center text-sm font-semibold text-gray-700 uppercase tracking-wider">Health Score</th>
                        <th class="px-4 py-2 border text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Positive Summary</th>
                        <th class="px-4 py-2 border text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Negative Summary</th>
                        <th class="px-4 py-2 border text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Reasoning</th>
                        <th class="px-4 py-2 border text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Headlines</th>
                        <th class="px-4 py-2 border text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Last Updated (UTC)</th>
                    </tr>
                </thead>
                <tbody class="text-gray-700">
    """

    for row in data:
        headlines_list = json.loads(row.get('headlines', '[]'))
        headlines_html = "<ul>" + "".join([f"<li class='text-xs ml-4 list-disc'>{h}</li>" for h in headlines_list]) + "</ul>" if headlines_list else "<span>-</span>"
        
        # Format last_updated timestamp
        last_updated_str = "-"
        if updated_ts := row.get('last_updated'):
             try:
                 dt_obj = datetime.fromisoformat(updated_ts.replace('Z', '+00:00')) # Ensure timezone awareness
                 last_updated_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
             except ValueError:
                 last_updated_str = updated_ts # Show raw if parsing fails

        html_content += f"""
                    <tr class="hover:bg-gray-50">
                        <td class="border px-4 py-2 font-semibold text-blue-600">
                             <a href="https://finance.yahoo.com/quote/{row.get('ticker','_')}" target="_blank" rel="noopener noreferrer" title="View on Yahoo Finance">{row.get('ticker','?')}</a>
                        </td>
                        <td class="border px-4 py-2 text-center">{f"{row.get('model_confidence_win', ''):.3f}" if row.get('model_confidence_win') is not None else '-'}</td>
                        {generate_score_cell(row.get('bullishness_score'))}
                        {generate_score_cell(row.get('news_sentiment_score'))}
                        {generate_score_cell(row.get('growth_score'))}
                        {generate_score_cell(row.get('valuation_score'))}
                        {generate_score_cell(row.get('financial_health_score'))}
                        <td class="border px-4 py-2 text-sm">{row.get('positive_summary','-')}</td>
                        <td class="border px-4 py-2 text-sm">{row.get('negative_summary','-')}</td>
                        <td class="border px-4 py-2 text-sm">{row.get('reasoning','-')}</td>
                        <td class="border px-4 py-2 text-sm">{headlines_html}</td>
                        <td class="border px-4 py-2 text-xs text-gray-500 whitespace-nowrap">{last_updated_str}</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
    """

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully generated HTML report: {output_filename}")
    except IOError as e:
        print(f"Error writing HTML file {output_filename}: {e}")

# --- Main Execution ---
def setup_arg_parser():
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(description="Generate a static HTML report from the stock analysis SQLite database.")
    parser.add_argument('-db', '--database', default=DEFAULT_DB_FILENAME,
                        help=f"Path to the SQLite database file (default: {DEFAULT_DB_FILENAME}).")
    parser.add_argument('-o', '--output', default=DEFAULT_HTML_FILENAME,
                        help=f"Path for the output HTML file (default: {DEFAULT_HTML_FILENAME}).")
    parser.add_argument('-d', '--date', default=None,
                        help="Specific analysis date (YYYY-MM-DD) to generate report for (default: most recent).")
    return parser

if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if not os.path.exists(args.database):
        print(f"Error: Database file not found at '{args.database}'")
        sys.exit(1)

    analysis_date_to_use = args.date
    if not analysis_date_to_use:
        analysis_date_to_use = get_most_recent_date(args.database)
        if not analysis_date_to_use:
            print(f"Error: Could not determine the most recent analysis date from '{args.database}'. Use --date YYYY-MM-DD.")
            sys.exit(1)
        else:
            print(f"Using most recent analysis date found in DB: {analysis_date_to_use}")
    else:
        # Optional: Validate date format
        try:
             datetime.strptime(analysis_date_to_use, '%Y-%m-%d')
             print(f"Using specified analysis date: {analysis_date_to_use}")
        except ValueError:
             print(f"Error: Invalid date format for --date. Please use YYYY-MM-DD.")
             sys.exit(1)


    report_data = fetch_analysis_data(args.database, analysis_date_to_use)

    if report_data:
        generate_html_report(report_data, analysis_date_to_use, args.output)
    else:
        print(f"No analysis data found for date {analysis_date_to_use} in the database.")