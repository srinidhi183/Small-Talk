# evaluation/batch_tester.py
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Type hint for the app instance to avoid circular imports
if 'SmallTalkApp' not in locals():
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from evaluation.main import SmallTalkApp

def run_all_modes_and_save_to_excel(
    app: 'SmallTalkApp',
    keywords_list: List[str],
    prompt_version: str = "v01"
) -> str:
    """
    Runs ALL supported modes for each keyword and saves the consolidated results 
    into a single, detailed, multi-sheet Excel file.
    """
    print(f"--- Starting Comprehensive Batch Test (All Modes to Excel) ---")
    
    # 1. Prepare data holders for the report sheets
    responses_data = []      # For Sheet 2: LLM Responses
    detailed_metrics = []    # For Sheet 3: Detailed Metrics
    all_evaluations = []     # Helper list to aggregate data for Sheet 1

    start_time = datetime.now()
    total_test_cases = len(keywords_list)
    
    mode_functions = {
        'search': app.run_search,
        'trending': app.run_trending,
        'recommendation': app.run_recommendation,
        'update': app.run_update
    }

    # 2. Loop through each keyword from the input list
    for idx, keyword in enumerate(keywords_list, start=1):
        print(f"\n--- Running Test Case ID: {idx}, Keywords: '{keyword}' ---")
        case_start_time = time.time()
        
        # Fetch a shared context once per keyword for fair comparison
        context = app._get_shared_context(keyword)
        
        # Loop through each mode for the current keyword
        for mode_name, run_function in mode_functions.items():
            print(f"  -> Running mode: '{mode_name}'")
            result = run_function(keywords=keyword, retrieval_context=context)
            mode_duration = time.time() - case_start_time
            
            # Append data for Sheet 2: LLM Responses
            responses_data.append({
                'ID': idx,
                'Keywords': keyword,
                'Mode': mode_name, # Add mode to distinguish rows
                'LLM Response (JSON)': json.dumps(result.get('data', {}), indent=2),
                'Source Articles/News': "\n---\n".join(context) if context else "No context fetched."
            })
            
            # Append data for Sheet 3: Detailed Metrics
            all_evals_for_case = {**result.get('evaluation_p1', {}), **result.get('evaluation_p2', {})}
            for metric, metric_data in all_evals_for_case.items():
                detailed_metrics.append({
                    'ID': idx,
                    'Keywords': keyword,
                    'Mode': mode_name, # Add mode here as well
                    'Metric': metric,
                    'Score': metric_data.get('score'),
                    'Reason': metric_data.get('reason', ''),
                    'Time (sec)': round(mode_duration, 2)
                })
            
            all_evaluations.append(all_evals_for_case)

    # 3. Aggregate data for Sheet 1: General Information
    metrics_p1 = app.pipeline1.metrics if app.pipeline1 else {}
    metrics_p2 = app.pipeline2.metrics if app.pipeline2 else {}
    all_metric_objects = {**metrics_p1, **metrics_p2}
    passed_counts = {m: 0 for m in all_metric_objects}
    total_scores = {m: 0.0 for m in all_metric_objects}
    valid_counts = {m: 0 for m in all_metric_objects}

    for eval_case in all_evaluations:
        for metric, data in eval_case.items():
            if metric in all_metric_objects and data.get('score') is not None:
                total_scores[metric] += data['score']
                valid_counts[metric] += 1
                if data.get('successful', False):
                    passed_counts[metric] += 1
    avg_scores = {m: total_scores[m] / valid_counts[m] if valid_counts[m] > 0 else 0.0 for m in total_scores}

    general_info_data = {
        'Metric Name': [name for name in all_metric_objects],
        'Threshold': [getattr(obj, 'threshold', 'N/A') for obj in all_metric_objects.values()],
        'Passed Count': [passed_counts[name] for name in all_metric_objects],
        'Average Score': [avg_scores[name] for name in all_metric_objects]
    }
    
    summary_header = pd.DataFrame([{"Date": start_time.strftime("%Y-%m-%d"), "Time": start_time.strftime("%H:%M:%S"),
                                    "Number of Test Cases": total_test_cases, "Output Type Tested": "all_modes",
                                    "Prompt Version": prompt_version}])

    # 4. Create DataFrames for each sheet
    df_general_header = summary_header
    df_general_metrics = pd.DataFrame(general_info_data)
    df_responses = pd.DataFrame(responses_data)
    df_detailed = pd.DataFrame(detailed_metrics)

    # 5. Write all data to the Excel report
    report_filename = f"report_all_modes_{start_time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(report_filename, engine='xlsxwriter') as writer:
        df_general_header.to_excel(writer, sheet_name='General Information', index=False, startrow=0)
        df_general_metrics.to_excel(writer, sheet_name='General Information', index=False, startrow=len(df_general_header)+1)
        df_responses.to_excel(writer, sheet_name='LLM Responses', index=False)
        df_detailed.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # Auto-adjust column widths for readability
        for sheet_name, df_sheet in [('General Information', df_general_metrics), ('LLM Responses', df_responses), ('Detailed Metrics', df_detailed)]:
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df_sheet.columns):
                max_len = max(df_sheet[col].astype(str).map(len).max(), len(str(col))) + 2
                worksheet.set_column(idx, idx, max_len)

    print(f"--- Comprehensive Batch Test Finished. Report saved to: {report_filename} ---")
    return report_filename

def run_test_from_cli_keywords(
    app: 'SmallTalkApp',
    cli_keywords: List[str],
    output_type: str,
    prompt_version: str = "v01"
) -> str:
    """This function is for generating a report for a *single* mode."""
    # (Your existing logic for this function)
    pass

def run_batch_test_from_file(
    app: 'SmallTalkApp',
    excel_file_path: str,
    output_type: str,
    prompt_version: str = "v01"
) -> str:
    """This function is for generating a report for a *single* mode from a file."""
    # (Your existing logic for this function)
    pass
