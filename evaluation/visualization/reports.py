import os
from utils import get_directory_manager, get_model_names
from datetime import datetime


def generate_final_summary_report(results, overfitting_analyzer, data):
    print(f"\n{'=' * 80}")
    print("📋 FINAL SUMMARY REPORT")
    print(f"{'=' * 80}")

    model_names = get_model_names()

    try:
        dir_manager = get_directory_manager()
        output_dir = dir_manager.results_dir
    except:
        output_dir = os.path.join("output", "results")
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"final_summary_report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)

    report_content = []
    report_content.append("=" * 80)
    report_content.append("WINE QUALITY CLASSIFICATION - FINAL SUMMARY REPORT")
    report_content.append("=" * 80)
    report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")

    best_model_content = generate_best_model_summary(results)
    report_content.extend(best_model_content)

    report_content.append("")
    report_content.append("✅ All analyses complete!")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))

        print(f"\n💾 Final summary report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️ Could not save report to file: {e}")

    for line in report_content:
        print(line)

    return report_path


def generate_best_model_summary(results):
    """Generate summary of best performing model"""
    content = []
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    content.append("🏆 BEST PERFORMING MODEL")
    content.append("-" * 30)
    content.append(f"Model: {best_model[0]}")
    content.append(f"Accuracy: {best_model[1]['accuracy']:.4f}")
    content.append(f"F1 Score: {best_model[1]['f1']:.4f}")
    content.append(f"Precision: {best_model[1]['precision']:.4f}")
    content.append(f"Recall: {best_model[1]['recall']:.4f}")
    content.append("")

    print(f"🏆 Best performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

    return content