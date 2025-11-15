"""
Export utilities for John Lewis Christmas Ad NLP Analysis.
Handles data export in multiple formats with comprehensive metadata.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import zipfile
import logging

from src.config import OUTPUT_DIR, EXPORT_CONFIG

class DataExporter:
    """
    Comprehensive data exporter for analysis results.
    Supports multiple formats and includes detailed metadata.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the data exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = output_dir or OUTPUT_DIR / "exports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def export_to_csv(self, data: pd.DataFrame, filename: str, 
                     include_metadata: bool = True) -> str:
        """
        Export DataFrame to CSV with optional metadata.
        
        Args:
            data: DataFrame to export
            filename: Output filename
            include_metadata: Whether to include metadata columns
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}.csv"
        
        # Prepare data for export
        export_data = data.copy()
        
        if not include_metadata:
            # Remove internal/metadata columns
            metadata_columns = [col for col in export_data.columns 
                              if col.startswith('_') or 'internal' in col.lower()]
            export_data = export_data.drop(columns=metadata_columns)
        
        # Export to CSV
        export_data.to_csv(
            output_path, 
            index=False, 
            encoding=EXPORT_CONFIG['encoding']
        )
        
        self.logger.info(f"Exported {len(export_data)} rows to {output_path}")
        return str(output_path)
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame], 
                       filename: str) -> str:
        """
        Export multiple DataFrames to Excel with multiple sheets.
        
        Args:
            data_dict: Dictionary of sheet_name -> DataFrame
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                # Clean sheet name for Excel
                clean_sheet_name = str(sheet_name)[:31]  # Excel limit
                df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
        
        self.logger.info(f"Exported Excel file with {len(data_dict)} sheets to {output_path}")
        return str(output_path)
    
    def export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to JSON format.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}.json"
        
        # Convert pandas objects to serializable format
        json_data = self._prepare_for_json(data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Exported JSON data to {output_path}")
        return str(output_path)
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def export_analysis_summary(self, results: Dict[str, Any]) -> str:
        """
        Export comprehensive analysis summary.
        
        Args:
            results: Pipeline results dictionary
            
        Returns:
            Path to exported summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive summary
        summary = {
            "analysis_metadata": {
                "timestamp": timestamp,
                "analysis_type": "John Lewis Christmas Ad NLP Analysis",
                "completed_steps": list(results.keys()),
                "total_comments_analyzed": len(results.get('preprocessing', pd.DataFrame()))
            },
            "sentiment_analysis": {},
            "topic_modeling": {},
            "demographic_insights": {},
            "methodology": {
                "traditional_topic_modeling": "TF-IDF + LDA",
                "transformer_topic_modeling": "BERTopic with Sentence-BERT",
                "sentiment_analysis": "TextBlob + RoBERTa transformer",
                "demographic_analysis": "Linguistic pattern analysis (proxies)"
            }
        }
        
        # Add sentiment summary
        if 'sentiment_analysis' in results:
            sentiment_data = results['sentiment_analysis']['combined_results']
            sentiment_summary = results['sentiment_analysis']['sentiment_summary']
            
            summary['sentiment_analysis'] = {
                "total_comments": len(sentiment_data),
                "sentiment_distribution": sentiment_data['consensus_sentiment'].value_counts().to_dict(),
                "average_polarity": float(sentiment_data['textblob_polarity'].mean()),
                "average_subjectivity": float(sentiment_data['textblob_subjectivity'].mean()),
                "detailed_summary": sentiment_summary.to_dict('records')
            }
        
        # Add topic modeling summary
        if 'traditional_topics' in results:
            traditional_summary = results['traditional_topics']['topic_summary']
            summary['topic_modeling']['traditional_lda'] = {
                "number_of_topics": len(traditional_summary),
                "topics": traditional_summary.to_dict('records')
            }
        
        if 'bertopic' in results:
            bertopic_summary = results['bertopic']['topic_summary']
            summary['topic_modeling']['bertopic'] = {
                "number_of_topics": len(bertopic_summary),
                "topics": bertopic_summary.to_dict('records')
            }
        
        # Add demographic insights
        if 'demographic_analysis' in results:
            demographic_data = results['demographic_analysis']['demographic_data']
            
            summary['demographic_insights'] = {
                "total_comments": len(demographic_data),
                "linguistic_age_groups": demographic_data['age_group_proxy'].value_counts().to_dict(),
                "expressiveness_levels": demographic_data['expressiveness'].value_counts().to_dict(),
                "average_formality_score": float(demographic_data['formality_score'].mean())
            }
        
        return self.export_to_json(summary, "comprehensive_analysis_summary")
    
    def export_all_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Export all analysis results in multiple formats.
        
        Args:
            results: Pipeline results dictionary
            
        Returns:
            Dictionary mapping export types to file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        
        try:
            # Export processed data
            if 'preprocessing' in results:
                exported_files['processed_data'] = self.export_to_csv(
                    results['preprocessing'], 
                    "processed_comments",
                    include_metadata=False
                )
            
            # Export sentiment analysis
            if 'sentiment_analysis' in results:
                exported_files['sentiment_analysis'] = self.export_to_csv(
                    results['sentiment_analysis']['combined_results'],
                    "sentiment_analysis_results",
                    include_metadata=False
                )
            
            # Export traditional topics
            if 'traditional_topics' in results:
                exported_files['traditional_topics'] = self.export_to_csv(
                    results['traditional_topics']['topic_summary'],
                    "traditional_topic_modeling",
                    include_metadata=False
                )
            
            # Export BERTopic results
            if 'bertopic' in results:
                exported_files['bertopic_results'] = self.export_to_csv(
                    results['bertopic']['topic_summary'],
                    "bertopic_results",
                    include_metadata=False
                )
            
            # Export demographic analysis
            if 'demographic_analysis' in results:
                exported_files['demographic_analysis'] = self.export_to_csv(
                    results['demographic_analysis']['demographic_data'],
                    "demographic_insights",
                    include_metadata=False
                )
            
            # Export Excel summary
            excel_data = {}
            if 'preprocessing' in results:
                excel_data['Processed_Data'] = results['preprocessing']
            if 'sentiment_analysis' in results:
                excel_data['Sentiment_Results'] = results['sentiment_analysis']['combined_results']
            if 'traditional_topics' in results:
                excel_data['Traditional_Topics'] = results['traditional_topics']['topic_summary']
            if 'bertopic' in results:
                excel_data['BERTopic_Results'] = results['bertopic']['topic_summary']
            
            if excel_data:
                exported_files['excel_summary'] = self.export_to_excel(
                    excel_data, 
                    "complete_analysis_summary"
                )
            
            # Export comprehensive JSON summary
            exported_files['json_summary'] = self.export_analysis_summary(results)
            
            # Create zip archive of all exports
            zip_path = self.output_dir / f"analysis_exports_{timestamp}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for export_type, file_path in exported_files.items():
                    zipf.write(file_path, f"{export_type}_{Path(file_path).name}")
            
            exported_files['zip_archive'] = str(zip_path)
            
            self.logger.info(f"Successfully exported all results. Files saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during export: {e}")
            raise
        
        return exported_files
    
    def create_sample_export(self, results: Dict[str, Any], sample_size: int = 100) -> str:
        """
        Create a sample export with limited data for sharing.
        
        Args:
            results: Pipeline results dictionary
            sample_size: Number of samples to include
            
        Returns:
            Path to sample export file
        """
        sample_data = {}
        
        # Sample processed data
        if 'preprocessing' in results:
            processed_data = results['preprocessing']
            sample_data['sample_comments'] = processed_data.sample(
                min(sample_size, len(processed_data))
            )
        
        # Sample sentiment results
        if 'sentiment_analysis' in results:
            sentiment_data = results['sentiment_analysis']['combined_results']
            sample_data['sample_sentiment'] = sentiment_data.sample(
                min(sample_size, len(sentiment_data))
            )
        
        # Export sample data
        return self.export_to_excel(sample_data, "sample_analysis_data")

def main():
    """
    Example usage of data export utilities.
    """
    # Create sample results
    sample_results = {
        'preprocessing': pd.DataFrame({
            'text': ['Sample comment 1', 'Sample comment 2'],
            'processed_text': ['sample comment 1', 'sample comment 2'],
            'sentiment_text': ['Sample comment 1', 'Sample comment 2']
        }),
        'sentiment_analysis': {
            'combined_results': pd.DataFrame({
                'text': ['Sample comment 1', 'Sample comment 2'],
                'consensus_sentiment': ['positive', 'neutral'],
                'textblob_polarity': [0.5, 0.0]
            }),
            'sentiment_summary': pd.DataFrame({
                'Metric': ['Test'],
                'Positive': ['50%'],
                'Neutral': ['50%']
            })
        }
    }
    
    # Initialize exporter
    exporter = DataExporter()
    
    # Export all results
    exported_files = exporter.export_all_results(sample_results)
    
    print("Export completed!")
    for export_type, file_path in exported_files.items():
        print(f"{export_type}: {file_path}")

if __name__ == "__main__":
    main()
