import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
from Cleaning import *
#Run
#streamlit run "D:\Data Analysis\Projects\Weather Data(2000-2023)\Dashboard.py"

# ===================== CONFIG =====================
st.set_page_config(page_title="🌦️ Weather Analytics Dashboard", layout="wide")

# ===================== DATA LOADER =====================
class WeatherData:
    def __init__(self, df, coords_df):
        self.df = df
        self.coords_df = coords_df

    def apply_filters(self):
        """Apply Streamlit sidebar filters to DataFrame"""
        st.sidebar.header("🔎 Filters")

        continents = st.sidebar.multiselect(
            "🌍 Select Continent",
            self.df['Continent'].unique(),
            default=self.df['Continent'].unique()
        )
        self.df = self.df[self.df['Continent'].isin(continents)]

        countries = st.sidebar.multiselect(
            "🏳 Select Country",
            self.df['Country'].unique(),
            default=self.df['Country'].unique()
        )
        self.df = self.df[self.df['Country'].isin(countries)]

        min_date, max_date = self.df['Date'].min(), self.df['Date'].max()
        date_range = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date])
        if len(date_range) == 2:
            self.df = self.df[
                (self.df['Date'] >= pd.to_datetime(date_range[0])) &
                (self.df['Date'] <= pd.to_datetime(date_range[1]))
            ]
        
        # Season Filter   
        season_filter = st.sidebar.multiselect(
            "Select Season", self.df['Season'].unique(), default=self.df['Season'].unique()
        )
        self.df = self.df[self.df['Season'].isin(season_filter)]

        return self.df

# ===================== ANALYTICS =====================
class WeatherAnalytics:
    
    @staticmethod
    def extract_weather_insights(df):
        
        # --- Weather data & recommendations ---
        data = [
            ["Nigeria", "Summer", "12.29 mm heavy rain",
              "✅ Strengthen drainage systems\n✅ Plan infrastructure projects outside peak rainy periods\n✅ Alert farmers to waterlogging risks"],
            ["India", "Summer", "10.51 mm heavy rain",
              "✅ Prepare flood-prone areas with sandbags\n✅ Promote water-resistant crops\n✅ Maintain transport routes"],
            ["Indonesia", "Most of Year", "Heavy rain",
              "✅ Improve flood management year-round\n✅ Invest in rainwater harvesting\n✅ Ensure emergency readiness"],
            ["South Africa", "Any", "Wind speed 19–29 km/h",
              "✅ Secure outdoor infrastructure\n✅ Adjust aviation/marine schedules\n✅ Warn farmers about pesticide spraying"],
            ["United Kingdom", "Any", "Wind speed 19–29 km/h",
              "✅ Secure infrastructure\n✅ Share wind advisories\n✅ Adjust transport operations"],
            ["Argentina", "Any", "Wind speed 19–29 km/h",
              "✅ Secure outdoor infrastructure\n✅ Adjust schedules for ports & airports"],
            ["India", "Spring", "39 °C high temperature",
              "✅ Issue heatwave alerts\n✅ Promote cooling shelters\n✅ Adjust school/work timings"],
            ["Myanmar", "Spring", "35 °C high temperature",
              "✅ Hydration campaigns\n✅ Protect vulnerable groups\n✅ Plan agricultural irrigation"]
        ]
        
        df = pd.DataFrame(data, columns=["Country", "Season", "Condition", "Recommendations"])
        
        # --- Streamlit UI ---
        st.set_page_config(page_title="Weather Risk Recommendations", layout="wide")
        
        st.title("🌦 Weather Risk & Recommendations Dashboard")
        st.markdown("This dashboard summarizes key weather observations and actionable recommendations.")
        
        # Table display
        st.dataframe(df, use_container_width=True)
        
        # Optionally add filters
        st.subheader("🔍 Filter by Country or Season")
        country = st.selectbox("Select Country", ["All"] + sorted(df["Country"].unique().tolist()))
        season = st.selectbox("Select Season", ["All"] + sorted(df["Season"].unique().tolist()))
        
        filtered_df = df.copy()
        if country != "All":
            filtered_df = filtered_df[filtered_df["Country"] == country]
        if season != "All":
            filtered_df = filtered_df[filtered_df["Season"] == season]
        
        st.write("### 📋 Filtered Recommendations")
        st.dataframe(filtered_df, use_container_width=True)
        

# ===================== VISUALIZATION =====================
class WeatherPlots:
    feature_colors = {
        "Temp_Max": "#ff7f50",
        "Temp_Min": "#1e90ff",
        "Precipitation_Sum": "#00cc66",
        "Windspeed_Max": "#cc66ff",
        "Windgusts_Max": "#9933cc",
        "Sunshine_Duration": "#ffcc00"
    }

    @classmethod
    def plot_feature(cls, df, group_col, feature, kind="bar", rot=90):
        data = df.groupby(group_col)[feature].mean().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#2a2a2a")

        data.plot(
            ax=ax,
            kind=kind,
            rot=rot,
            color=cls.feature_colors.get(feature, "#ffffff"),
            edgecolor="white"
        )

        ax.set_ylabel(feature.replace("_", " "), color="white")
        ax.set_xlabel(group_col, color="white")
        ax.tick_params(colors="white")
        ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        plt.tight_layout()
        return fig
    
    @classmethod
    def plot_correlation_heatmap(self, corr):
        facecolor = "#1e1e1e"
        text_color = "white"
        cmap = "coolwarm"

        fig, ax = plt.subplots(figsize=(7, 6), facecolor=facecolor)
        ax.set_facecolor(facecolor)

        # Heatmap
        sns.heatmap(
            (corr)*100,
            annot=True,
            fmt=".2f",  # تقريب القيم لـ 2 أرقام عشرية
            cmap=cmap,
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlation"}
        )
        
        ax.set_title("📊 Correlation Heatmap", fontsize=16, color=text_color, fontweight="bold")
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)

        #  Tick labels
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_color(text_color)

        plt.tight_layout()
        return fig

# ===================== MAP BUILDER =====================
class WeatherMaps:
        
    @staticmethod
    def build_temp_map(df, coords_df, column, title=""):
        
        df_grouped = df.groupby("Country", as_index=False).agg(
                Avg_Temp=(column, "mean"),
                Min_Temp=(column, "min"),
                Max_Temp=(column, "max")
        )
        merged_df = pd.merge(df_grouped, coords_df, on="Country", how="left")    
        # 🎯 Define your own classes
        def classify_temp(temp):
            if temp < 10:
                return "Very Cold  < 10 °C"
            elif temp < 20:
                return "Cold       < 20 °C"
            elif temp < 30:
                return "Warm       < 30 °C"
            elif temp < 35:
                return "Hot        < 35 °C"
            else:
                return "Very Hot   > 35 °C"

        merged_df["Class"] = merged_df["Avg_Temp"].apply(classify_temp)

        # 🎨 Fixed colors
        colors = {
            "Very Cold  < 10 °C": "darkblue",
            "Cold       < 20 °C": "blue",
            "Warm       < 30 °C": "yellow",
            "Hot        < 35 °C": "red",
            "Very Hot   > 35 °C": "darkred"
        }

        # Normalize values ​​for heatmap
        min_temp = merged_df["Avg_Temp"].min()
        max_temp = merged_df["Avg_Temp"].max()
        merged_df["NormTemp"] = (merged_df["Avg_Temp"] - min_temp) / (max_temp - min_temp)

        # Create Map
        m = folium.Map(location=[20, 0], zoom_start=2)

        heat_data = merged_df.dropna(subset=["Latitude", "Longitude"])[
            ["Latitude", "Longitude", "NormTemp"]
        ].values.tolist()
        
        HeatMap(heat_data, radius=25, blur=15,max_zoom=4).add_to(m)

        # Tooltip
        for _, row in merged_df.iterrows():
            if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                tooltip_text = (
                    f"<b>{row['Country']}</b><br>"
                    f"🌡 Average: {row['Avg_Temp']:.2f}°C<br>"
                    f"⬆️ Max: {row['Max_Temp']:.2f}°C<br>"
                    f"⬇️ Min: {row['Min_Temp']:.2f}°C"
                )
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=6,
                    color=colors[row["Class"]],
                    fill=True,
                    fill_color=colors[row["Class"]],
                    fill_opacity=0.8,
                    tooltip=tooltip_text
                ).add_to(m)
        # -------------------- 6️⃣  Legend  --------------------
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 235px;
            background-color: white;
            border: 2px solid #333;
            z-index: 9999;
            font-size: 14px;
            padding: 10px;
            border-radius: 8px;
            color: black;
            font-weight: 600;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
        <b>{title}</b><br>
        """
        for label, color in colors.items():
            legend_html += f'<i style="background:{color};width:15px;height:15px;float:left;margin-right:8px;"></i> {label}<br>'
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    @staticmethod
    def build_precipitation_map_by_condition(df, coords_df, column="Precipitation_Sum", title="Average Precipitation"):
        
        df_grouped = df.groupby("Country", as_index=False).agg(
                Avg_Precip=(column, "mean"),
                Min_Precip=(column, "min"),
                Max_Precip=(column, "max")
        )
        merged_df = pd.merge(df_grouped, coords_df, on="Country", how="left") 
        # 🎯 Define precipitation classes
        def classify_precip(p):
            if p == 0:
                return "No Rain       0 (mm)"
            elif p <= 10:
                return "Light Rain    < 10 (mm)"
            elif p <= 30:
                return "Moderate Rain < 30 (mm)"
            elif p <= 70:
                return "Heavy Rain    < 70 (mm)"
            else:
                return "Very Heavy Rain > 70 (mm)"

        merged_df["Precipitation_Class"] = merged_df["Avg_Precip"].apply(classify_precip)

        # 🎨 Fixed colors for classes
        class_colors = {
            "No Rain       0 (mm)": "#ffffcc",
            "Light Rain    < 10 (mm)": "#a1dab4",
            "Moderate Rain < 30 (mm)": "#41b6c4",
            "Heavy Rain    < 70 (mm)": "#225ea8",
            "Very Heavy Rain > 70 (mm)": "#0c2c84"
        }

        # 🌡 Normalize precipitation for HeatMap
        min_p = merged_df["Avg_Precip"].min()
        max_p = merged_df["Avg_Precip"].max()
        merged_df["NormPrecip"] = (merged_df["Avg_Precip"] - min_p) / (max_p - min_p)

        # 🗺 Create map
        m = folium.Map(location=[20, 0], zoom_start=2)

        # 🔥 HeatMap layer
        heat_data = merged_df.dropna(subset=["Latitude", "Longitude"])[
            ["Latitude", "Longitude", "NormPrecip"]
        ].values.tolist()

        HeatMap(heat_data, radius=25, blur=15, max_zoom=4).add_to(m)

        # 🟢 Circle markers with tooltips
        for _, row in merged_df.iterrows():
            if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                tooltip_text = (
                    f"<b>{row['Country']}</b><br>"
                    f"🌧 Avg: {row['Avg_Precip']:.2f} mm<br>"
                    f"⬆️ Max: {row['Max_Precip']:.2f} mm<br>"
                    f"⬇️ Min: {row['Min_Precip']:.2f} mm"
                )
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=6,
                    color=class_colors[row["Precipitation_Class"]],
                    fill=True,
                    fill_color=class_colors[row["Precipitation_Class"]],
                    fill_opacity=0.8,
                    tooltip=tooltip_text
                ).add_to(m)

        # 🔖 Legend
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 235px;
            background-color: white;
            border: 2px solid #333;
            z-index: 9999;
            font-size: 14px;
            padding: 10px;
            border-radius: 8px;
            color: black;
            font-weight: 600;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
        <b>{title}</b><br>
        """
        for label, color in class_colors.items():
            legend_html += f'<i style="background:{color};width:15px;height:15px;float:left;margin-right:8px;"></i> {label}<br>'
        legend_html += "</div>"

        m.get_root().html.add_child(folium.Element(legend_html))

        return m

# ===================== Pivot Table =====================
class WeatherPivotTable:
    
    cols_value=['Temp_Max','Temp_Min','Precipitation_Sum','Windspeed_Max','Windgusts_Max','Sunshine_Duration']
    col_colors = {
        "Temp_Max": lambda ratio: f"background-color:rgb({200+55*ratio:.0f},50,50); color:white;",  # أحمر للأعلى
        "Temp_Min": lambda ratio: f"background-color:rgb(50,50,{200+55*ratio:.0f}); color:white;",  # أزرق للأعلى
        "Precipitation_Sum": lambda ratio: f"background-color:rgb(50,{150+105*ratio:.0f},50); color:white;",  # أخضر للأمطار
        "Windspeed_Max": lambda ratio: f"background-color:rgb({150+100*ratio:.0f},50,{150+100*ratio:.0f}); color:white;",  # بنفسجي
        "Windgusts_Max": lambda ratio: f"background-color:rgb({150+100*ratio:.0f},50,{150+100*ratio:.0f}); color:white;",  # بنفسجي
        "Sunshine_Duration": lambda ratio: f"background-color:rgb(255,{220+35*ratio:.0f},{150-50*ratio:.0f}); color:black;",  # أصفر برتقالي
    }
    
    @classmethod    
    def get_color_by_grade(cls, df, value, col, col_min_max):
        min_val, max_val = col_min_max[col]
        if pd.isna(value):
            return ""
        ratio = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

        # Assign grade
        if ratio   < 0.2: grade = 0
        elif ratio < 0.4: grade = 1
        elif ratio < 0.6: grade = 2
        elif ratio < 0.8: grade = 3
        else            : grade = 4

        bg_color = cls.col_colors[col][grade]

        # --- Determine text color (black/white) based on brightness ---
        # Convert HEX to RGB
        r = int(bg_color[1:3], 16)
        g = int(bg_color[3:5], 16)
        b = int(bg_color[5:7], 16)
        # Perceived brightness formula
        brightness = (0.299*r + 0.587*g + 0.114*b)
        text_color = "#000000" if brightness > 160 else "#FFFFFF"

        return f"background-color:{bg_color}; color:{text_color}; font-weight:bold;"

    @classmethod
    def styled_pivot_table_with_totals(cls, df, index,
                                        group_column=None,
                                        max_height=500,
                                        border_css="3px solid black",
                                        blank_repeats=True,
                                        add_subtotals=None,
                                        aggfunc="mean"):

        pivot = df.pivot_table(index=index, values=cls.cols_value, aggfunc=aggfunc)
        pivot = pivot.apply(lambda col: col.round(2) if pd.api.types.is_numeric_dtype(col) else col)
        pivot_reset = pivot.reset_index()

        if add_subtotals:
            if isinstance(add_subtotals, str):
                add_subtotals = [add_subtotals]
            subtotal_rows = []
            for group_col in add_subtotals:
                for group_val, group_df in pivot_reset.groupby(group_col, sort=False):
                    subtotal = pd.DataFrame({col: "" for col in pivot_reset.columns}, index=[0])
                    subtotal.loc[0, group_col] = f"TOTAL ({group_val})"
                    for v in cls.cols_value:
                        subtotal.loc[0, v] = round(group_df[v].agg(aggfunc), 2)
                    subtotal_rows.append((group_df.index.max() + 0.1, subtotal))
            for idx, subtotal in sorted(subtotal_rows, key=lambda x: x[0], reverse=True):
                pivot_reset = pd.concat([pivot_reset.iloc[:int(idx)+1], subtotal, pivot_reset.iloc[int(idx)+1:]])
            pivot_reset = pivot_reset.reset_index(drop=True)

        if blank_repeats:
            for col in index:
                pivot_reset[col] = pivot_reset[col].mask(pivot_reset[col] == pivot_reset[col].shift(1), "")

        def highlight_values(val, col):
            if col not in cls.col_colors or not isinstance(val, (int, float)):
                return ""
            col_min = pivot_reset[col].replace("", float("nan")).min()
            col_max = pivot_reset[col].replace("", float("nan")).max()
            ratio = 0 if col_max == col_min else (val - col_min) / (col_max - col_min)
            return cls.col_colors[col](ratio)

        numeric_cols = pivot_reset.select_dtypes(include="number").columns
        styled = (pivot_reset.style
                  .format({col: "{:.2f}" for col in numeric_cols})
                  .apply(lambda row: [highlight_values(v, col) for col, v in row.items()], axis=1)
                  .set_table_styles([
                      {"selector": "th",
                        "props": "background-color:#333; color:white; font-weight:bold; text-align:center; position:sticky; top:0; z-index:2;"},
                      {"selector": "td", "props": "text-align:center; padding:6px; font-weight:500;"},
                      {"selector": "tbody tr:nth-child(even)", "props": "background-color:#f5f5f5;"},
                      {"selector": "tbody tr:nth-child(odd)", "props": "background-color:#ffffff;"}
                  ])
                  .hide(axis="index"))

        st.markdown(f"""
        <div style="
            max-height:{max_height}px;
            overflow-y:auto;
            border:1px solid #ccc;
            border-radius:8px;
            background:white;
        ">
            <style>
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    color: #222;
                    font-size: 14px;
                }}
                th {{
                    position: sticky;
                    top: 0;
                    background-color: #333;
                    color: white;
                    font-size: 15px;
                }}
            </style>
            {styled.to_html()}
        </div>
        """, unsafe_allow_html=True)
        
    @classmethod    
    def render_feature_legends(cls):
        st.markdown("#### 📊 Color Legends")
        for feature, color_func in cls.col_colors.items():
            st.markdown(f"**{feature}**")
            html = "<div style='display:flex; margin-bottom:10px;'>"
            labels = ["Very Low", "Low", "Medium", "High", "Very High"]

            ratios = [0.2, 0.4, 0.6, 0.8, 1]
            for ratio, label in zip(ratios, labels):
                color_style = color_func(ratio).split(";")[0].replace("background-color:", "")
                html += f"<div style='flex:1; background-color:{color_style}; text-align:center; padding:4px; font-size:14px; border:1px solid #ccc;'>{label}</div>"

            st.markdown(html, unsafe_allow_html=True)

    
# ===================== DASHBOARD =====================
class WeatherDashboard:
    def __init__(self, df, coords_df):
        self.df = df
        self.coords_df = coords_df

    def show_kpis(self):
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🔥 Hottest Country", self.df.groupby('Country')['Temp_Mean'].mean().idxmax())
        with col2:
            st.metric("🌡️ Max Temp (°C)", self.df['Temp_Max'].max())
        with col3:
            st.metric("🌧️ Wettest Country", self.df.groupby('Country')['Precipitation_Sum'].mean().idxmax())
        with col4:
            st.metric("🌧️ Max Precipitation", self.df['Precipitation_Sum'].max())
        with col5:
            st.metric("🌡️ Avg Max Temp (°C)", round(self.df['Temp_Max'].mean(), 2))

    def render_tabs(self):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📄 Overview", "📈 Visualizations", "🗺️ Maps", "📊 Pivot Table", "🧠 Insights"]
        )

        with tab1:
            st.markdown("### Dataset Overview")
            st.dataframe(self.df.sample(10))
            st.markdown("### Descriptive Statistics")
            st.dataframe(self.df.describe())

        with tab2:
            st.markdown("## 📈 Temperature Trend")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌡️ Average Max Temperature per Month")
                fig = WeatherPlots.plot_feature(self.df, "Month", "Temp_Max", rot=0)
                st.pyplot(fig)

            with col2:
                st.markdown("#### 📈 Maximum Temperature Trend Over Time")
                fig = WeatherPlots.plot_feature(self.df, "Year", "Temp_Max")
                st.pyplot(fig)
                
            st.markdown("## 🌧 Rainfall Analysis")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("### 🌧️ Average Rainfall per Month")
                fig = WeatherPlots.plot_feature(self.df, "Month", "Precipitation_Sum", rot=0)
                st.pyplot(fig)
                
            with col4:
                st.markdown("#### 🌧 Average Rainfall per Country")
                fig = WeatherPlots.plot_feature(self.df, "Country", "Precipitation_Sum", rot=75)
                st.pyplot(fig)
            
            st.markdown("## Correlation & Windspeed")
            col5, col6 = st.columns(2) 
            with col5:
                st.markdown("#### Windspeed for Countries")
                fig = WeatherPlots.plot_feature(self.df, "Country", "Windspeed_Max", rot=75)
                st.pyplot(fig)
                
            with col6:
                st.markdown("#### Windgusts for Countries")
                fig = WeatherPlots.plot_feature(self.df, "Country", "Windgusts_Max", rot=75)
                st.pyplot(fig)
            
            st.markdown("### Correlation Heatmap")        
            corr = self.df[['Temp_Max', 'Temp_Min', 'Precipitation_Sum','Windspeed_Max','Windgusts_Max','Sunshine_Duration']].corr()
            fig = WeatherPlots.plot_correlation_heatmap(corr)
            st.pyplot(fig)

        with tab3:
            wm = WeatherMaps()
            
            st.markdown("### 🌡 Heatmap for Average Maximum Temperature")
            max_map = wm.build_temp_map(df=self.df, coords_df=self.coords_df, column="Temp_Max", title="Max Temp")
            st_folium(max_map, width=1200, height=500, key="max_map")
            
            st.markdown("### 🌡 Heatmap for Average Minimum Temperature")
            min_map = wm.build_temp_map(df=self.df, coords_df=self.coords_df, column="Temp_Min", title="Min Temp")
            st_folium(min_map, width=1200, height=500, key="min_map")
            
            st.markdown("### 🌡 Heatmap for Average Precipitation")   
            precipitation_map = wm.build_precipitation_map_by_condition(df=self.df, coords_df=self.coords_df)
            st_folium(precipitation_map, width=1200, height=500, key="precipitation_map")
            
        with tab4:
            
            st.title("🌍 Climate Summary Dashboard")
            WeatherPivotTable.render_feature_legends()
            
            WeatherPivotTable.styled_pivot_table_with_totals(
                self.df,
                index=['Continent', 'Country', 'Season'],
                add_subtotals='Continent',
                aggfunc='mean',
                blank_repeats=True
            )

        with tab5:
            st.markdown("## 🧠 Key Insights & Recommendations")
            WeatherAnalytics.extract_weather_insights(self.df)
         
# ===================== MAIN APP =====================
def main():
    st.title("🌦️ Weather Analytics Dashboard")
    st.caption("Explore climate trends interactively.")

    data = WeatherData(df, coords_df)
    filtered_df = data.apply_filters()

    dashboard = WeatherDashboard(filtered_df, coords_df)
    dashboard.show_kpis()
    dashboard.render_tabs()

if __name__ == "__main__":
    main()
