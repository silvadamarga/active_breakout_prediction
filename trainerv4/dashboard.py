#!/usr/bin/env python3
"""
Stock Analysis Static Site Generator

This script reads data from the 'stock_analysis_v2.db' SQLite database
(created by the analysis script) and generates a single, self-contained
static HTML file ('index.html') to display the latest analysis for each ticker.

The generated HTML includes Tailwind CSS and JavaScript for styling, sorting,
and filtering the data table.
"""

import sqlite3
import html
from datetime import datetime, timezone
import logging
import sys

# --- Configuration ---
class Config:
    """Holds configuration for the generator script."""
    ANALYSIS_DB_FILENAME = "stock_analysis_v2.db"
    OUTPUT_HTML_FILENAME = "index.html"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_score_color(score):
    """Returns Tailwind CSS classes based on a 1-10 score."""
    if not isinstance(score, (int, float)):
        return "bg-gray-100 text-gray-800"
    if score <= 3:
        return "bg-red-100 text-red-800"
    if score <= 6:
        return "bg-yellow-100 text-yellow-800"
    return "bg-green-100 text-green-800"

def fetch_latest_analysis_data():
    """
    Fetches the latest analysis for each ticker from the database.
    
    Returns:
        list: A list of dictionaries, where each dict is a stock's data.
              Returns None if an error occurs.
    """
    logging.info(f"Connecting to database: {Config.ANALYSIS_DB_FILENAME}")
    
    # This query joins companies with their *most recent* analysis entry.
    # It's ordered by bullishness score (desc) and then confidence (desc).
    query = """
    SELECT
        c.ticker,
        c.company_name,
        c.sector,
        da.analysis_date,
        da.model_confidence_win,
        da.bullishness_score,
        da.reasoning,
        da.news_sentiment_score,
        da.growth_score,
        da.valuation_score,
        da.financial_health_score,
        da.positive_summary,
        da.negative_summary
    FROM companies c
    JOIN daily_analysis da ON c.ticker = da.ticker
    INNER JOIN (
        SELECT ticker, MAX(analysis_date) as max_date
        FROM daily_analysis
        GROUP BY ticker
    ) latest ON da.ticker = latest.ticker AND da.analysis_date = latest.max_date
    ORDER BY da.bullishness_score DESC, da.model_confidence_win DESC;
    """
    
    try:
        with sqlite3.connect(Config.ANALYSIS_DB_FILENAME) as conn:
            conn.row_factory = sqlite3.Row # Allows accessing columns by name
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                logging.warning("No data found in the database. Was the analysis script run?")
                return []
                
            logging.info(f"Fetched {len(rows)} records from the database.")
            return [dict(row) for row in rows]
            
    except sqlite3.OperationalError as e:
        logging.error(f"Database error: {e}")
        logging.error("Please ensure 'stock_analysis_v2.db' exists and the analysis script has been run.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data fetching: {e}")
        return None

def generate_html_table_rows(stock_data):
    """Generates HTML <tr> elements for each stock."""
    rows_html = []
    if not stock_data:
        return '<tr><td colspan="12" class="px-4 py-4 text-center text-gray-500">No stock data found.</td></tr>'

    for stock in stock_data:
        # --- Helper function for safe HTML escaping ---
        def e(text):
            return html.escape(str(text)) if text is not None else "N/A"

        # --- Get color classes for scores ---
        bullish_color = get_score_color(stock['bullishness_score'])
        news_color = get_score_color(stock['news_sentiment_score'])
        growth_color = get_score_color(stock['growth_score'])
        val_color = get_score_color(stock['valuation_score'])
        health_color = get_score_color(stock['financial_health_score'])

        # --- Format data ---
        ticker_link = f'https://finance.yahoo.com/quote/{e(stock["ticker"])}'
        confidence_pct = f"{stock['model_confidence_win']:.1%}" if isinstance(stock['model_confidence_win'], float) else e(stock['model_confidence_win'])

        # --- Create the table row ---
        rows_html.append(f"""
        <tr class="border-b border-gray-200 bg-white hover:bg-gray-50 text-sm">
            <td class="px-4 py-3 font-medium text-blue-600 whitespace-nowrap">
                <a href="{ticker_link}" target="_blank" rel="noopener noreferrer" class="hover:underline">{e(stock["ticker"])}</a>
            </td>
            <td class="px-4 py-3 text-gray-700">{e(stock["company_name"])}</td>
            <td class="px-4 py-3 text-gray-600">{e(stock["sector"])}</td>
            <td class="px-4 py-3 text-gray-600 whitespace-nowrap">{e(stock["analysis_date"])}</td>
            <td class_base="px-3 py-2 font-mono font-bold text-center rounded-md {bullish_color}">{e(stock["bullishness_score"])}</td>
            <td class_base="px-3 py-2 font-mono text-center rounded-md {news_color}">{e(stock["news_sentiment_score"])}</td>
            <td class_base="px-3 py-2 font-mono text-center rounded-md {growth_color}">{e(stock["growth_score"])}</td>
            <td class_base="px-3 py-2 font-mono text-center rounded-md {val_color}">{e(stock["valuation_score"])}</td>
            <td class_base="px-3 py-2 font-mono text-center rounded-md {health_color}">{e(stock["financial_health_score"])}</td>
            <td class="px-4 py-3 text-gray-700 min-w-[20rem] max-w-sm"><p class="truncate" title="{e(stock['reasoning'])}">{e(stock["reasoning"])}</p></td>
            <td class="px-4 py-3 text-gray-700 min-w-[20rem] max-w-sm"><p class="truncate" title="{e(stock['positive_summary'])}">{e(stock["positive_summary"])}</p></td>
            <td class="px-4 py-3 text-gray-700 min-w-[20rem] max-w-sm"><p class="truncate" title="{e(stock['negative_summary'])}">{e(stock["negative_summary"])}</p></td>
            <td class="px-4 py-3 font-medium text-gray-800 whitespace-nowrap">{confidence_pct}</td>
        </tr>
        """.replace('class_base', 'class')) # Small hack to format multi-line f-strings
    
    return "\n".join(rows_html)

def generate_full_html(table_rows_html, generation_time):
    """Wraps the table rows in a full HTML5 document."""
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-R-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom styles for table sorting */
        th.sortable {{
            position: relative;
            cursor: pointer;
            padding-right: 1.5rem; /* Space for arrow */
        }}
        th.sortable::after,
        th.sortable::before {{
            position: absolute;
            right: 0.5rem;
            opacity: 0.3;
            content: '';
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
        }}
        th.sortable::before {{
            bottom: 50%; /* Top arrow */
            border-bottom: 5px solid currentColor;
        }}
        th.sortable::after {{
            top: 50%; /* Bottom arrow */
            border-top: 5px solid currentColor;
        }}
        th.sort-asc::before,
        th.sort-desc::after {{
            opacity: 1;
        }}
        
        /* Simple sticky header for the table */
        thead th {{
            position: sticky;
            top: 0;
            z-index: 10;
        }}
    </style>
</head>
<body class="bg-gray-100 font-sans">

    <div class="container mx-auto p-4 md:p-8">
        <header class="mb-6">
            <h1 class="text-3xl font-bold text-gray-900">Stock Analysis Dashboard</h1>
            <p class="text-sm text-gray-600">Last generated: {generation_time}</p>
        </header>

        <!-- Filter Input -->
        <div class="mb-4">
            <input type="text" id="filterInput"
                   class="w-full max-w-md p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                   placeholder="Filter by Ticker, Name, or Sector...">
        </div>

        <!-- Main Table Container -->
        <div class="overflow-x-auto bg-white rounded-lg shadow-md">
            <table class="min-w-full divide-y divide-gray-200" id="stockTable">
                <thead class="bg-gray-50">
                    <tr>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="0" data-type="string">Ticker</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="1" data-type="string">Company Name</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="2" data-type="string">Sector</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="3" data-type="string">Date</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="4" data-type="number">Bullish</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="5" data-type="number">News</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="6" data-type="number">Growth</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="7" data-type="number">Val</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="8" data-type="number">Health</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reasoning</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Positive</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Negative</th>
                        <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sortable" data-column="12" data-type="percent">Confidence</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200" id="tableBody">
                    {table_rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const filterInput = document.getElementById('filterInput');
            const tableBody = document.getElementById('tableBody');
            const table = document.getElementById('stockTable');
            const headers = table.querySelectorAll('th.sortable');
            const rows = Array.from(tableBody.querySelectorAll('tr'));

            // --- Filter Function ---
            filterInput.addEventListener('keyup', () => {{
                const filterText = filterInput.value.toLowerCase();
                rows.forEach(row => {{
                    // Check Ticker (col 0), Name (col 1), Sector (col 2)
                    const ticker = row.cells[0].textContent.toLowerCase();
                    const name = row.cells[1].textContent.toLowerCase();
                    const sector = row.cells[2].textContent.toLowerCase();
                    const isVisible = ticker.includes(filterText) || 
                                      name.includes(filterText) || 
                                      sector.includes(filterText);
                    row.style.display = isVisible ? '' : 'none';
                }});
            }});

            // --- Sort Function ---
            headers.forEach(header => {{
                header.addEventListener('click', () => {{
                    const column = parseInt(header.dataset.column, 10);
                    const type = header.dataset.type;
                    const isAsc = header.classList.contains('sort-asc');
                    const direction = isAsc ? -1 : 1;

                    // Reset other headers
                    headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                    header.classList.toggle('sort-asc', !isAsc);
                    header.classList.toggle('sort-desc', isAsc);

                    // Sort the rows array
                    const sortedRows = rows.sort((a, b) => {{
                        let valA = a.cells[column].textContent.trim();
                        let valB = b.cells[column].textContent.trim();

                        if (type === 'number') {{
                            valA = parseFloat(valA) || 0;
                            valB = parseFloat(valB) || 0;
                        }} else if (type === 'percent') {{
                            valA = parseFloat(valA.replace('%', '')) || 0;
                            valB = parseFloat(valB.replace('%', '')) || 0;
                        }}
                        // Default is string compare
                        
                        if (valA < valB) return -1 * direction;
                        if (valA > valB) return 1 * direction;
                        return 0;
                    }});

                    // Re-append rows to the table body
                    sortedRows.forEach(row => tableBody.appendChild(row));
                }});
            }});
        }});
    </script>

</body>
</html>
    """

def save_html_to_file(html_content):
    """Saves the generated HTML content to the output file."""
    try:
        with open(Config.OUTPUT_HTML_FILENAME, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"Successfully generated static site: {Config.OUTPUT_HTML_FILENAME}")
    except IOError as e:
        logging.error(f"Failed to write HTML file: {e}")
        sys.exit(1)

def main():
    """Main function to orchestrate the site generation."""
    stock_data = fetch_latest_analysis_data()
    
    if stock_data is None:
        logging.error("Halting generation due to database error.")
        sys.exit(1)
        
    if not stock_data:
        logging.warning("Database was empty, generating an empty dashboard.")

    table_rows_html = generate_html_table_rows(stock_data)
    
    generation_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    full_html = generate_full_html(table_rows_html, generation_time)
    
    save_html_to_file(full_html)

if __name__ == "__main__":
    main()
