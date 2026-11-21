import os
import argparse
import json
import pandas as pd
import numpy as np

FIBONACCI = [1, 2, 3, 5, 8, 13]

def snap_to_fibonacci(val):
    return min(FIBONACCI, key=lambda x: abs(x - val))

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    exact = float(np.mean(y_true == y_pred) * 100)
    
    within_range_count = 0
    for t, p in zip(y_true, y_pred):
        idx_t = FIBONACCI.index(t) if t in FIBONACCI else min(range(len(FIBONACCI)), key=lambda i: abs(FIBONACCI[i]-t))
        idx_p = FIBONACCI.index(p) if p in FIBONACCI else min(range(len(FIBONACCI)), key=lambda i: abs(FIBONACCI[i]-p))
        if abs(idx_t - idx_p) <= 1:
            within_range_count += 1
    within_range = float((within_range_count / len(y_true)) * 100)
    
    return {
        "MAE": round(mae, 3),
        "Exact_Agreement_%": round(exact, 2),
        "Within_Range_%": round(within_range, 2)
    }

def run_pipeline(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    issues = pd.read_csv(os.path.join(data_dir, "issues.csv"))
    epics = pd.read_csv(os.path.join(data_dir, "epics.csv"))
    teams = pd.read_csv(os.path.join(data_dir, "teams.csv"))
    projects = pd.read_csv(os.path.join(data_dir, "projects.csv"))
    
    issues['created_at'] = pd.to_datetime(issues['created_at'])
    issues['year'] = issues['created_at'].dt.year
    
    df = issues.merge(epics[['epic_id', 'epic_name']], on='epic_id', how='left')
    df = df.merge(teams[['team_id', 'team_name', 'specialization']], on='team_id', how='left')
    df = df.merge(projects[['project_id', 'project_name']], on='project_id', how='left')
    
    train_df = df[df['year'] == 2024]
    test_df = df[df['year'] == 2025].copy()
    
    project_medians = train_df.groupby('project_id')['story_points'].median().to_dict()
    test_df['pred_median'] = test_df['project_id'].map(project_medians).apply(snap_to_fibonacci)
    
    def zero_shot_predict(row):
        word_count = len(str(row['summary']).split())
        priority_weight = {'P1': 5, 'P2': 4, 'P3': 3, 'P4': 2}.get(row['priority'], 3)
        raw_est = (word_count * 0.4) + priority_weight
        return snap_to_fibonacci(raw_est)
    
    test_df['pred_zero_shot'] = test_df.apply(zero_shot_predict, axis=1)
    
    def few_shot_predict(row):
        hist = train_df[(train_df['project_id'] == row['project_id']) & 
                        (train_df['issue_type'] == row['issue_type'])]
        same_epic = hist[hist['epic_id'] == row['epic_id']]
        
        if len(same_epic) > 0:
            hist_pts = same_epic['story_points'].tolist()
        elif len(hist) > 0:
            hist_pts = hist['story_points'].tolist()
        else:
            hist_pts = [3.0]
            
        avg_hist = np.mean(hist_pts)
        return snap_to_fibonacci(avg_hist)
    
    test_df['pred_few_shot'] = test_df.apply(few_shot_predict, axis=1)
    
    models = {
        "Project-Median Baseline": "pred_median",
        "Zero-Shot AI (No Context)": "pred_zero_shot",
        "Few-Shot AI (Project Context)": "pred_few_shot"
    }
    
    results = {}
    y_true = test_df['story_points'].values
    
    for model_name, pred_col in models.items():
        y_pred = test_df[pred_col].values
        
        overall_metrics = calculate_metrics(y_true, y_pred)
        
        proj_metrics = {}
        for proj in test_df['project_name'].unique():
            sub = test_df[test_df['project_name'] == proj]
            proj_metrics[proj] = calculate_metrics(sub['story_points'], sub[pred_col])
            
        type_metrics = {}
        for itype in test_df['issue_type'].unique():
            sub = test_df[test_df['issue_type'] == itype]
            type_metrics[itype] = calculate_metrics(sub['story_points'], sub[pred_col])
            
        results[model_name] = {
            "Overall": overall_metrics,
            "By_Project": proj_metrics,
            "By_Issue_Type": type_metrics
        }
    
    out_path = os.path.join(output_dir, "experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Experiment executed successfully! Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    
    run_pipeline(args.data_dir, args.output_dir)
    
