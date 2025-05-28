import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from preprocessing import load_data, preprocess_data, get_important_features
from model import train_model, save_model, load_model, predict_price, plot_feature_importance
from utils import (
    plot_price_distribution, plot_price_vs_feature,
    plot_correlation_matrix, plot_prediction_vs_actual,
    get_neighborhood_stats, plot_neighborhood_prices,
    format_price
)

# Cấu hình trang
st.set_page_config(
    page_title="Dự báo Giá Nhà",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }

    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }

    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
    }

    .feature-section {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }

    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
''', unsafe_allow_html=True)

def main():
    # Tiêu đề chính
    st.markdown('<p class="main-header">🏠 ỨNG DỤNG DỰ BÁO GIÁ NHÀ</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/cottage.png", width=80)
    st.sidebar.title("Điều hướng")

    # Menu điều hướng
    page = st.sidebar.radio(
        "Chọn trang:",
        ["Dự báo giá", "Phân tích dữ liệu", "Thông tin mô hình"]
    )

    # Đường dẫn dữ liệu
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    model_path = "models/saved_model.pkl"

    # Tải dữ liệu
    @st.cache_data
    def get_data():
        train_df = load_data(train_path)
        return train_df

    train_df = get_data()

    # Tiền xử lý dữ liệu để lấy numerical_cols và categorical_cols
    X, y, preprocessor, numerical_cols, categorical_cols = preprocess_data(train_df)

    # Kiểm tra và tạo mô hình
    if not os.path.exists(model_path) or (os.path.exists(model_path) and os.path.getsize(model_path) == 0):
        with st.spinner('Đang huấn luyện mô hình... Vui lòng đợi trong giây lát...'):
            os.makedirs('models', exist_ok=True)
            model_results = train_model(X, y, preprocessor, model_type='random_forest')
            model = model_results['model']
            save_model(model, preprocessor, model_path)
    else:
        model, preprocessor = load_model(model_path)

    # Phần Dự báo giá
    if page == "Dự báo giá":
        st.markdown('<p class="sub-header">Nhập thông tin ngôi nhà</p>', unsafe_allow_html=True)

        # Tạo layout 3 cột cho form nhập liệu
        col1, col2, col3 = st.columns(3)

        # Thông tin cơ bản - Cột 1
        with col1:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.subheader("Thông tin cơ bản")

            neighborhoods = sorted(train_df['Neighborhood'].unique().tolist())
            neighborhood = st.selectbox("Khu vực", neighborhoods,
                                        index=neighborhoods.index('NAmes') if 'NAmes' in neighborhoods else 0)

            mszoning_values = train_df['MSZoning'].unique().tolist()
            mszoning = st.selectbox("Phân vùng", mszoning_values,
                                    index=mszoning_values.index('RL') if 'RL' in mszoning_values else 0)

            year_built = st.slider("Năm xây dựng", int(train_df['YearBuilt'].min()), 2025, 2000)
            year_remod = st.slider("Năm cải tạo", int(train_df['YearRemodAdd'].min()), 2025, 2000)

            overall_qual = st.slider("Chất lượng tổng thể (1-10)", 1, 10, 7)
            overall_cond = st.slider("Điều kiện tổng thể (1-10)", 1, 10, 5)
            st.markdown('</div>', unsafe_allow_html=True)

        # Thông số diện tích - Cột 2
        with col2:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.subheader("Thông số diện tích")

            lot_area = st.number_input("Diện tích lô đất (sq.ft)", min_value=1000, max_value=100000, value=8500)
            total_bsmt_sf = st.number_input("Diện tích tầng hầm (sq.ft)", min_value=0, max_value=6000, value=1000)
            gr_liv_area = st.number_input("Diện tích sinh hoạt (sq.ft)", min_value=500, max_value=6000, value=1500)
            first_flr_sf = st.number_input("Diện tích tầng 1 (sq.ft)", min_value=500, max_value=4000, value=1000)
            second_flr_sf = st.number_input("Diện tích tầng 2 (sq.ft)", min_value=0, max_value=4000, value=500)
            wood_deck_sf = st.number_input("Diện tích sàn gỗ (sq.ft)", min_value=0, max_value=1000, value=100)
            st.markdown('</div>', unsafe_allow_html=True)

        # Các tiện ích - Cột 3
        with col3:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.subheader("Các tiện ích")

            bedrooms = st.slider("Số phòng ngủ", 0, 10, 3)
            bathrooms = st.slider("Số phòng tắm đầy đủ", 0, 5, 2)
            half_bath = st.slider("Số phòng tắm nửa", 0, 3, 0)
            kitchen_qual_values = train_df['KitchenQual'].unique().tolist()
            kitchen_qual = st.selectbox("Chất lượng nhà bếp", kitchen_qual_values,
                                        index=kitchen_qual_values.index('Gd') if 'Gd' in kitchen_qual_values else 0)

            garage_type_values = list(train_df['GarageType'].unique())
            if None in garage_type_values:
                garage_type_values.remove(None)
            garage_type = st.selectbox("Loại garage", garage_type_values, index=garage_type_values.index(
                'Attchd') if 'Attchd' in garage_type_values else 0)

            garage_cars = st.slider("Sức chứa xe trong garage", 0, 5, 2)
            garage_area = st.number_input("Diện tích garage (sq.ft)", min_value=0, max_value=1500, value=500)
            st.markdown('</div>', unsafe_allow_html=True)

        # Nút dự đoán
        predict_btn = st.button("DỰ ĐOÁN GIÁ NHÀ", key="predict_btn")

        if predict_btn:
            # Tạo dictionary các đặc trưng
            features = {
                'Neighborhood': neighborhood,
                'MSZoning': mszoning,
                'YearBuilt': year_built,
                'YearRemodAdd': year_remod,
                'OverallQual': overall_qual,
                'OverallCond': overall_cond,
                'LotArea': lot_area,
                'TotalBsmtSF': total_bsmt_sf,
                'GrLivArea': gr_liv_area,
                '1stFlrSF': first_flr_sf,
                '2ndFlrSF': second_flr_sf,
                'WoodDeckSF': wood_deck_sf,
                'BedroomAbvGr': bedrooms,
                'FullBath': bathrooms,
                'HalfBath': half_bath,
                'KitchenQual': kitchen_qual,
                'GarageType': garage_type,
                'GarageCars': garage_cars,
                'GarageArea': garage_area,
                'TotRmsAbvGrd': bedrooms + 3,
                'Fireplaces': 1,
                'BsmtFullBath': 1,
                'OpenPorchSF': 50,
                'MasVnrArea': 100,
                'Exterior1st': 'VinylSd',
                'Exterior2nd': 'VinylSd',
                'BsmtFinSF1': 500,
                'SaleType': 'WD',
                'MasVnrType': 'BrkFace',
                'BsmtQual': 'TA',
                'BsmtCond': 'TA',
                'BsmtExposure': 'No',
                'BsmtFinType1': 'GLQ',
                'HeatingQC': 'Ex',
                'CentralAir': 'Y',
                'Electrical': 'SBrkr',
                'Functional': 'Typ',
                'FireplaceQu': 'Gd',
                'GarageFinish': 'Fin',
                'GarageQual': 'TA',
                'GarageCond': 'TA',
                'PavedDrive': 'Y',
                'SaleCondition': 'Normal',
                'BsmtUnfSF': 500,
                'BsmtFinSF2': 0,
                'LowQualFinSF': 0,
                'EnclosedPorch': 0,
                '3SsnPorch': 0,
                'ScreenPorch': 0,
                'PoolArea': 0,
                'MiscVal': 0,
                'MoSold': 6,
                'YrSold': 2025,
                'Alley': 'None',
                'PoolQC': 'None',
                'Fence': 'None',
                'MiscFeature': 'None'
            }

            # Chuyển features thành DataFrame
            features_df = pd.DataFrame([features])

            # Kiểm tra và thêm các cột thiếu
            for col in numerical_cols:
                if col not in features_df.columns:
                    features_df[col] = 0
            for col in categorical_cols:
                if col not in features_df.columns:
                    features_df[col] = 'None'

            # Đảm bảo thứ tự cột khớp với dữ liệu huấn luyện
            try:
                features_df = features_df.reindex(columns=X.columns, fill_value=0)
                st.write(f"Shape of features_df after reindex: {features_df.shape}")
            except ValueError as e:
                st.error(f"Lỗi khi tái lập chỉ số cột: {str(e)}")
                return

            # Thực hiện dự đoán
            try:
                # Transform features
                X_processed = preprocessor.transform(features_df)
                st.write(f"Shape of X_processed before prediction: {X_processed.shape}")
                # Ensure X_processed is 2D
                if len(X_processed.shape) > 2:
                    X_processed = X_processed.reshape(X_processed.shape[0], -1)
                    st.write(f"Shape of X_processed after reshape: {X_processed.shape}")
                # Make prediction directly
                price_prediction = model.predict(X_processed)[0]
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {str(e)}")
                return

            # Hiển thị kết quả dự đoán
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<p>Giá dự đoán của ngôi nhà là:</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="prediction-value">{format_price(price_prediction)}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Hiển thị thông tin so sánh
            st.markdown('<p class="sub-header">So sánh với thị trường</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Tìm giá trung bình khu vực
                neighborhood_data = train_df[train_df['Neighborhood'] == neighborhood]
                avg_price_neighborhood = neighborhood_data['SalePrice'].mean()

                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Giá trung bình khu vực {neighborhood}:** {format_price(avg_price_neighborhood)}")
                diff_pct = (price_prediction - avg_price_neighborhood) / avg_price_neighborhood * 100
                st.markdown(
                    f"**So với giá trung bình khu vực:** {'tăng' if diff_pct > 0 else 'giảm'} {abs(diff_pct):.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                # Tìm giá trung bình toàn thành phố
                avg_price_all = train_df['SalePrice'].mean()

                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Giá trung bình toàn thành phố:** {format_price(avg_price_all)}")
                diff_pct_all = (price_prediction - avg_price_all) / avg_price_all * 100
                st.markdown(
                    f"**So với giá trung bình toàn thành phố:** {'tăng' if diff_pct_all > 0 else 'giảm'} {abs(diff_pct_all):.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            # Hiển thị biểu đồ phân phối giá trên khu vực
            st.subheader(f"Phân phối giá nhà ở khu vực {neighborhood}")
            fig = plot_price_vs_feature(train_df, 'Neighborhood', title=f"Giá nhà theo khu vực (điểm đỏ là dự đoán)")

            # Thêm điểm dự đoán vào biểu đồ
            fig.add_trace(
                go.Scatter(
                    x=[neighborhood],
                    y=[price_prediction],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Dự đoán của bạn'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    # Phần Phân tích dữ liệu
    elif page == "Phân tích dữ liệu":
        st.markdown('<p class="sub-header">Phân tích dữ liệu giá nhà</p>', unsafe_allow_html=True)

        # Tạo tabs
        tab1, tab2, tab3 = st.tabs(["Phân phối giá", "Phân tích theo khu vực", "Tương quan"])

        with tab1:
            st.subheader("Phân phối giá nhà")
            price_dist_fig = plot_price_distribution(train_df)
            st.plotly_chart(price_dist_fig, use_container_width=True)

            st.subheader("Phân phối giá theo diện tích")
            area_price_fig = plot_price_vs_feature(train_df, 'GrLivArea', title="Giá nhà theo diện tích sinh hoạt")
            st.plotly_chart(area_price_fig, use_container_width=True)

            st.subheader("Phân phối giá theo chất lượng")
            qual_price_fig = plot_price_vs_feature(train_df, 'OverallQual', title="Giá nhà theo chất lượng tổng thể")
            st.plotly_chart(qual_price_fig, use_container_width=True)

        with tab2:
            st.subheader("Phân tích giá nhà theo khu vực")
            neighborhood_fig = plot_neighborhood_prices(train_df)
            st.plotly_chart(neighborhood_fig, use_container_width=True)

            st.subheader("Thống kê chi tiết theo khu vực")
            neighborhood_stats = get_neighborhood_stats(train_df)

            # Định dạng giá trị để hiển thị
            formatted_stats = neighborhood_stats.copy()
            formatted_stats['mean'] = formatted_stats['mean'].apply(lambda x: f"${x:,.2f}")
            formatted_stats['median'] = formatted_stats['median'].apply(lambda x: f"${x:,.2f}")
            formatted_stats['std'] = formatted_stats['std'].apply(lambda x: f"${x:,.2f}")

            # Đổi tên cột
            formatted_stats.columns = ['Giá trung bình', 'Giá trung vị', 'Độ lệch chuẩn', 'Số lượng']

            st.dataframe(formatted_stats, use_container_width=True)

            # Biểu đồ phân phối giá theo khu vực
            st.subheader("Biểu đồ phân phối giá theo khu vực")
            selected_neighborhood = st.selectbox(
                "Chọn khu vực để xem phân phối giá",
                sorted(train_df['Neighborhood'].unique().tolist())
            )

            if selected_neighborhood:
                # Lọc dữ liệu cho khu vực được chọn
                neighborhood_data = train_df[train_df['Neighborhood'] == selected_neighborhood]

                fig = px.histogram(
                    neighborhood_data, x='SalePrice', nbins=30,
                    title=f'Phân phối giá nhà ở khu vực {selected_neighborhood}',
                    labels={'SalePrice': 'Giá nhà ($)', 'count': 'Số lượng'}
                )
                fig.update_layout(
                    xaxis_title='Giá nhà ($)',
                    yaxis_title='Số lượng',
                    template='plotly_white'
                )

                # Thêm giá trung bình và trung vị
                avg_price = neighborhood_data['SalePrice'].mean()
                median_price = neighborhood_data['SalePrice'].median()

                fig.add_vline(x=avg_price, line_dash="dash", line_color="red",
                              annotation_text=f"Trung bình: ${avg_price:,.2f}",
                              annotation_position="top right")

                fig.add_vline(x=median_price, line_dash="dash", line_color="green",
                              annotation_text=f"Trung vị: ${median_price:,.2f}",
                              annotation_position="top left")

                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Tương quan giữa các đặc trưng")

            # Chọn các đặc trưng quan trọng cho ma trận tương quan
            important_features = [
                'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                'FullBath', 'YearBuilt', 'YearRemodAdd', '1stFlrSF'
            ]

            corr_fig = plot_correlation_matrix(train_df, important_features)
            st.plotly_chart(corr_fig, use_container_width=True)

            # Cho phép người dùng chọn đặc trưng để phân tích
            st.subheader("Phân tích tương quan với giá")

            numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
            numerical_cols = [col for col in numerical_cols if col != 'Id' and col != 'SalePrice']

            selected_feature = st.selectbox(
                "Chọn đặc trưng để xem tương quan với giá",
                numerical_cols
            )

            if selected_feature:
                feature_fig = plot_price_vs_feature(train_df, selected_feature)
                st.plotly_chart(feature_fig, use_container_width=True)

                # Tính hệ số tương quan
                correlation = train_df[[selected_feature, 'SalePrice']].corr().iloc[0, 1]

                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Hệ số tương quan với giá:** {correlation:.4f}")

                if abs(correlation) > 0.7:
                    st.markdown("🟢 **Mức độ tương quan:** Rất mạnh")
                elif abs(correlation) > 0.5:
                    st.markdown("🟡 **Mức độ tương quan:** Mạnh")
                elif abs(correlation) > 0.3:
                    st.markdown("🟠 **Mức độ tương quan:** Trung bình")
                else:
                    st.markdown("🔴 **Mức độ tương quan:** Yếu")
                st.markdown('</div>', unsafe_allow_html=True)

    # Phần Thông tin mô hình
    elif page == "Thông tin mô hình":
        st.markdown('<p class="sub-header">Thông tin chi tiết về mô hình</p>', unsafe_allow_html=True)

        # Tạo tabs
        tab1, tab2 = st.tabs(["Đánh giá mô hình", "Đặc trưng quan trọng"])

        # Load mô hình và thực hiện đánh giá
        model, preprocessor = load_model(model_path)
        X, y, _, numerical_cols, categorical_cols = preprocess_data(train_df)
        X_processed = preprocessor.transform(X)
        predictions = model.predict(X_processed)

        with tab1:
            st.subheader("Đánh giá hiệu năng mô hình")

            # Tính các chỉ số đánh giá
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)

            # Hiển thị thông tin mô hình
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### Root Mean Squared Error (RMSE)")
                st.markdown(f"#### ${rmse:,.2f}")
                st.markdown("*Sai số trung bình dự đoán*")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### R² Score")
                st.markdown(f"#### {r2:.4f}")
                st.markdown("*Độ chính xác của mô hình (0-1)*")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### Loại mô hình")
                st.markdown("#### Random Forest")
                st.markdown("*Ensemble Learning Method*")
                st.markdown('</div>', unsafe_allow_html=True)

            # Biểu đồ so sánh giá dự đoán và thực tế
            st.subheader("So sánh giá dự đoán và thực tế")
            prediction_fig = plot_prediction_vs_actual(y, predictions)
            st.plotly_chart(prediction_fig, use_container_width=True)

            # Biểu đồ phân phối sai số
            error = y - predictions

            fig = px.histogram(
                x=error, nbins=50,
                title='Phân phối sai số dự đoán',
                labels={'x': 'Sai số ($)'}
            )
            fig.update_layout(
                xaxis_title='Sai số ($)',
                yaxis_title='Số lượng',
                template='plotly_white'
            )

            # Thêm đường giá trị trung bình của sai số
            fig.add_vline(x=error.mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Sai số trung bình: ${error.mean():,.2f}",
                          annotation_position="top right")

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Tầm quan trọng của các đặc trưng")

            if hasattr(model, 'feature_importances_'):
                # Get feature names after preprocessing
                feature_names = []
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    # Fallback for older scikit-learn versions
                    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

                # Create a DataFrame with feature importances
                importances = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                })
                importances = importances.sort_values('importance', ascending=False).head(20)

                # Create an interactive plot
                fig = px.bar(
                    importances, x='importance', y='feature',
                    orientation='h',
                    title='Top 20 đặc trưng quan trọng nhất',
                    labels={'importance': 'Mức độ quan trọng', 'feature': 'Đặc trưng'}
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # More detailed explanation
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### Giải thích mức độ quan trọng của đặc trưng")
                st.markdown("""
                **Mức độ quan trọng** của một đặc trưng trong mô hình Random Forest thể hiện mức độ ảnh hưởng của đặc trưng đó đến quyết định của mô hình.

                Các đặc trưng có mức độ quan trọng cao nhất thường là những yếu tố ảnh hưởng lớn nhất đến giá nhà, như:

                * **Chất lượng tổng thể** (OverallQual) - Đánh giá chất lượng tổng thể của ngôi nhà
                * **Diện tích sinh hoạt** (GrLivArea) - Tổng diện tích các không gian sinh hoạt
                * **Diện tích tầng hầm** (TotalBsmtSF) - Kích thước của tầng hầm
                * **Số chỗ đậu xe trong garage** (GarageCars) - Sức chứa của garage
                * **Vị trí khu vực** (Neighborhood) - Khu vực nơi ngôi nhà tọa lạc
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Mô hình này không cung cấp thông tin về tầm quan trọng của các đặc trưng.")

            # Giải thích ảnh hưởng của các đặc trưng quan trọng
            st.subheader("Phân tích chi tiết các đặc trưng quan trọng")

            important_features = [
                'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                'YearBuilt', 'Neighborhood'
            ]

            selected_feature = st.selectbox(
                "Chọn đặc trưng để phân tích chi tiết",
                important_features
            )

            if selected_feature:
                # Hiển thị biểu đồ quan hệ với giá
                feature_fig = plot_price_vs_feature(train_df, selected_feature)
                st.plotly_chart(feature_fig, use_container_width=True)

                # Giải thích ý nghĩa của đặc trưng
                explanations = {
                    'OverallQual': """
                        **Chất lượng tổng thể (OverallQual)** đánh giá chất lượng vật liệu và hoàn thiện của ngôi nhà trên thang điểm từ 1-10.

                        * Ngôi nhà với điểm chất lượng cao (8-10) thường có giá cao hơn đáng kể
                        * Mỗi 1 điểm tăng trong chất lượng có thể làm tăng giá trị nhà lên tới 20-30%
                        * Đây là một trong những yếu tố quan trọng nhất quyết định giá nhà
                    """,
                    'GrLivArea': """
                        **Diện tích sinh hoạt (GrLivArea)** là tổng diện tích các không gian sinh hoạt trên mặt đất của ngôi nhà.

                        * Diện tích lớn hơn thường đồng nghĩa với giá cao hơn
                        * Tương quan này khá tuyến tính - mỗi foot vuông thêm vào làm tăng giá trị nhà
                        * Tuy nhiên có ngưỡng giới hạn - nhà quá lớn so với khu vực xung quanh có thể không tăng giá trị tương ứng
                    """,
                    'GarageCars': """
                        **Sức chứa garage (GarageCars)** là số lượng xe hơi có thể đậu trong garage.

                        * Garage cho 2-3 xe thường được ưa chuộng nhất trên thị trường
                        * Garage lớn hơn (3+ xe) có thể làm tăng giá trị nhà, nhưng tỷ lệ tăng sẽ giảm dần
                        * Nhà không có garage thường có giá thấp hơn đáng kể so với nhà có garage
                    """,
                    'TotalBsmtSF': """
                        **Diện tích tầng hầm (TotalBsmtSF)** là tổng diện tích của tầng hầm.

                        * Tầng hầm hoàn thiện làm tăng giá trị sử dụng và giá bán của ngôi nhà
                        * Tầng hầm có thể tăng diện tích sinh hoạt mà không cần mở rộng diện tích xây dựng
                        * Chất lượng và mức độ hoàn thiện của tầng hầm ảnh hưởng lớn đến giá trị gia tăng
                    """,
                    'YearBuilt': """
                        **Năm xây dựng (YearBuilt)** cho biết tuổi của ngôi nhà.

                        * Nhà mới hơn thường có giá cao hơn do áp dụng công nghệ và vật liệu mới
                        * Nhà cũ có thể có giá trị về kiến trúc nhưng thường cần chi phí bảo trì cao hơn
                        * Các nhà xây trong một số giai đoạn nhất định có thể có giá trị lịch sử hoặc kiến trúc đặc biệt
                    """,
                    'Neighborhood': """
                        **Khu vực (Neighborhood)** cho biết vị trí địa lý của ngôi nhà.

                        * Là một trong những yếu tố quan trọng nhất quyết định giá nhà
                        * Khu vực có trường học tốt, an toàn, gần các tiện ích thường có giá cao hơn
                        * Sự khác biệt về giá giữa các khu vực có thể lên tới 200-300%
                        * Xu hướng phát triển của khu vực cũng ảnh hưởng đến giá trị nhà
                    """
                }

                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"### Giải thích đặc trưng: {selected_feature}")
                st.markdown(explanations.get(selected_feature, "Không có giải thích chi tiết cho đặc trưng này."))
                st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("© 2025 Ứng dụng Dự báo Giá Nhà | Phát triển bởi LanMo")

if __name__ == "__main__":
    main()