# Changelog - ProjekPID

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-21

### Added
- ✅ Complete ETL Pipeline (Extract, Transform, Load)
- ✅ Data extraction module with validation (`src/extract.py`)
- ✅ Data transformation with feature engineering (`src/transform.py`)
- ✅ Star schema database loading (`src/load.py`)
- ✅ Multiple ML models training (`src/train_model.py`)
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
- ✅ Batch prediction for future years (`src/predict.py`)
- ✅ Interactive Streamlit dashboard (`dashboard/app.py`)
- ✅ Configuration management (`config/config.yaml`)
- ✅ Comprehensive documentation (`USAGE.md`, `README.md`)
- ✅ Automated pipeline runner (`run_pipeline.py`)
- ✅ Quick start scripts for Windows & Linux (`run_pipeline.bat`, `run_pipeline.sh`)
- ✅ Utility functions (`src/utils.py`)
- ✅ Logging system
- ✅ .gitignore for Python projects

### Features
- **Data Cleaning**: Missing values, outliers, duplicates handling
- **Feature Engineering**: 
  - Productivity metrics
  - Lag features for time-series
  - Weather anomaly detection
  - Comfort index calculation
  - Categorical encoding
- **Model Evaluation**: R², RMSE, MAE, MAPE metrics
- **Cross-Validation**: Time-series 5-fold CV
- **Visualization**: 
  - Time series plots
  - Correlation heatmaps
  - Feature importance charts
  - Model comparison charts
  - Prediction vs Actual plots

### Database Schema
- Star Schema with:
  - Dimension Tables: `dim_provinsi`, `dim_waktu`
  - Fact Table: `fakta_produksi`

### Dashboard Pages
1. Overview - Dataset statistics
2. Time Series Analysis - Production trends
3. Correlation Analysis - Variable correlations
4. Weather Impact - Weather vs production
5. Province Comparison - Inter-province comparison
6. Geographic Visualization - Productivity map
7. Model Performance - ML model evaluation
8. Future Predictions - 5-year predictions

---

## Future Enhancements (Planned)

### [1.1.0] - Planned
- [ ] Google BigQuery integration
- [ ] Real-time data streaming
- [ ] API endpoint for predictions
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Unit tests & integration tests
- [ ] More advanced models (LSTM, Prophet)
- [ ] Weather data API integration
- [ ] Enhanced geographic visualization with actual maps
- [ ] Multi-language support (English/Indonesian)

### [1.2.0] - Planned
- [ ] Model interpretability (SHAP values)
- [ ] Automated hyperparameter tuning
- [ ] Model versioning system
- [ ] A/B testing framework
- [ ] Mobile-responsive dashboard
- [ ] Export reports to PDF
- [ ] Email notification system
- [ ] Data quality monitoring

---

## Bug Fixes

None reported yet.

---

## Contributors

- ProjekPID Team

---

**Note**: This project follows [Semantic Versioning](https://semver.org/).
