import os
import sys
import mimetypes

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector_logic import DeepfakeDetectorLogic

def get_file_type(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type:
        return mime_type
    # Fallback based on extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp']:
        return 'image/jpeg'
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return 'video/mp4'
    return 'application/octet-stream'

def evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    real_dir = os.path.join(dataset_dir, 'real')
    fake_dir = os.path.join(dataset_dir, 'fake')

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Dataset directories not found. Please create {real_dir} and {fake_dir}")
        return

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if not f.startswith('.')]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if not f.startswith('.')]

    if not real_files and not fake_files:
        print("No files found in dataset directories. Please add images/videos to backend/dataset/real and backend/dataset/fake.")
        return

    print("Initializing DeepfakeDetector...")
    detector = DeepfakeDetectorLogic()
    detector.load_models()
    print("Models loaded.")

    results = {
        'TP': 0, # Fake detected as Fake
        'TN': 0, # Real detected as Real
        'FP': 0, # Real detected as Fake
        'FN': 0, # Fake detected as Real
        'errors': 0
    }

    details_log = []

    def is_positive(verdict):
        # Returns True if detected as Fake/Suspicious
        return verdict in ["AI Generated", "Suspicious / Unverified"]

    print("\n--- Starting Evaluation ---\n")

    # Evaluate Real Files (Expecting Negative)
    print(f"Processing {len(real_files)} Real files...")
    for fpath in real_files:
        try:
            ftype = get_file_type(fpath)
            print(f"Analyzing {os.path.basename(fpath)}...")
            res = detector.analyze_local_file(fpath, ftype)
            verdict = res['verdict']
            score = res['score']
            
            if is_positive(verdict):
                results['FP'] += 1
                details_log.append(f"[FALSE POSITIVE] {os.path.basename(fpath)}: {verdict} (Score: {score})")
            else:
                results['TN'] += 1
                details_log.append(f"[CORRECT REAL] {os.path.basename(fpath)}: {verdict} (Score: {score})")
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            results['errors'] += 1

    # Evaluate Fake Files (Expecting Positive)
    print(f"\nProcessing {len(fake_files)} Fake files...")
    for fpath in fake_files:
        try:
            ftype = get_file_type(fpath)
            print(f"Analyzing {os.path.basename(fpath)}...")
            res = detector.analyze_local_file(fpath, ftype)
            verdict = res['verdict']
            score = res['score']
            
            if is_positive(verdict):
                results['TP'] += 1
                details_log.append(f"[CORRECT FAKE] {os.path.basename(fpath)}: {verdict} (Score: {score})")
            else:
                results['FN'] += 1
                details_log.append(f"[FALSE NEGATIVE] {os.path.basename(fpath)}: {verdict} (Score: {score})")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            results['errors'] += 1

    # Calculate Metrics
    total = results['TP'] + results['TN'] + results['FP'] + results['FN']
    accuracy = (results['TP'] + results['TN']) / total if total > 0 else 0
    precision = results['TP'] / (results['TP'] + results['FP']) if (results['TP'] + results['FP']) > 0 else 0
    recall = results['TP'] / (results['TP'] + results['FN']) if (results['TP'] + results['FN']) > 0 else 0

    # Generate HTML Report
    report_path = os.path.join(dataset_dir, 'evaluation_report.html')
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Veritas AI - Evaluation Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; color: #333; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .metric-card {{ background: #f1f5f9; padding: 20px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 2.5em; font-weight: bold; color: #3498db; }}
            .metric-label {{ color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }}
            tr:hover {{ background-color: #f1f5f9; }}
            .status-pass {{ color: #27ae60; font-weight: bold; }}
            .status-fail {{ color: #c0392b; font-weight: bold; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 500; }}
            .badge-real {{ background-color: #e8f5e9; color: #2e7d32; }}
            .badge-fake {{ background-color: #ffebee; color: #c62828; }}
            .badge-sus {{ background-color: #fff3e0; color: #ef6c00; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Veritas AI Evaluation Report</h1>
            <p>Generated on: {os.popen('date').read().strip()}</p>

            <h2>Summary Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{accuracy:.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{precision:.1%}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{recall:.1%}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">Total Files</div>
                </div>
            </div>

            <h2>Confusion Matrix</h2>
            <div class="metrics-grid">
                <div class="metric-card" style="background: #e8f5e9;">
                    <div class="metric-value" style="color: #2e7d32;">{results['TP']}</div>
                    <div class="metric-label">True Positives</div>
                    <small>Fake detected as Fake</small>
                </div>
                <div class="metric-card" style="background: #e8f5e9;">
                    <div class="metric-value" style="color: #2e7d32;">{results['TN']}</div>
                    <div class="metric-label">True Negatives</div>
                    <small>Real detected as Real</small>
                </div>
                <div class="metric-card" style="background: #ffebee;">
                    <div class="metric-value" style="color: #c62828;">{results['FP']}</div>
                    <div class="metric-label">False Positives</div>
                    <small>Real detected as Fake</small>
                </div>
                <div class="metric-card" style="background: #ffebee;">
                    <div class="metric-value" style="color: #c62828;">{results['FN']}</div>
                    <div class="metric-label">False Negatives</div>
                    <small>Fake detected as Real</small>
                </div>
            </div>

            <h2>Detailed Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Actual Type</th>
                        <th>Verdict</th>
                        <th>Score (Credibility)</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody>
    """

    import re
    for log in details_log:
        # Parse log string: "[STATUS] filename: Verdict (Score: X)"
        # Example: [FALSE POSITIVE] solid_real.jpg: AI Generated (Score: 10)
        match = re.match(r"\[(.*?)\] (.*?): (.*?) \(Score: (.*?)\)", log)
        
        if match:
            status_tag = match.group(1)
            filename = match.group(2)
            verdict = match.group(3)
            score_str = match.group(4)
            
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            
            actual_type = "Fake" if "FAKE" in status_tag or "FN" in status_tag or "CORRECT FAKE" in status_tag else "Real"
            if "CORRECT" in status_tag:
                row_class = "status-pass"
                result_text = "PASS"
            else:
                row_class = "status-fail"
                result_text = "FAIL"

            html_content += f"""
                    <tr>
                        <td>{filename}</td>
                        <td><span class="badge {'badge-fake' if actual_type == 'Fake' else 'badge-real'}">{actual_type}</span></td>
                        <td>{verdict}</td>
                        <td>{score:.1f}%</td>
                        <td class="{row_class}">{result_text}</td>
                    </tr>
            """
        else:
            html_content += f"<tr><td colspan='5'>{log}</td></tr>"

    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"\n--- Report Generated ---")
    print(f"HTML Report saved to: {report_path}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate()
