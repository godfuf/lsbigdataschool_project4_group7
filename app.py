from shiny import App, ui, render, reactive
from pathlib import Path
import platform
import matplotlib
import json
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
from htmltools import TagList, tags
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
WWW_DIR  = BASE_DIR / "www"

# ====== [1] 한글 폰트 설정 ======
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# ====== [2] 경상북도용 유틸리티 함수 ======
def unify_and_filter_region(df: pd.DataFrame, col: str, second_col: str = None) -> pd.DataFrame:
    """지역명을 통일하고 필터링하는 함수"""
    df = df.copy()

    # 시군구 단위 기준 정리
    region_keywords = [
        "포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", "문경시", "경산시",
        "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군",
        "울진군", "울릉군"
    ]

    pattern = "(" + "|".join(region_keywords) + ")"

    if second_col and second_col in df.columns:
        df['region_raw'] = df[col].astype(str).str.strip() + " " + df[second_col].astype(str).str.strip()
        df['region'] = df['region_raw'].str.extract(pattern)[0]
    else:
        df['region'] = df[col].astype(str).str.strip().str.extract(pattern)[0]

    # 군위군 제거
    return df[df['region'] != '군위군']

# ====== [3] 영천시용 거리 계산 함수 ======
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ====== [4] 경상북도 고정 데이터 ======
REGIONS = ["포항시","경주시","김천시","안동시","구미시","영주시","영천시","상주시","문경시","경산시",
           "의성군","청송군","영양군","영덕군","청도군","고령군","성주군","칠곡군","예천군","봉화군",
           "울진군","울릉군"]
POPULATION = [498296,257668,138999,154788,410306,99894,101185,93081,67674,285618,
              49336,23867,15494,34338,41641,32350,43543,111928,54868,28988,
              47872,9199]
pop_df = pd.DataFrame({'시군': REGIONS, '인구수': POPULATION})

# 22개 지역별 고유 색상 배정
REGION_COLORS = {
    "포항시": "#e6194b", "경주시": "#46f0f0", "김천시": "#ffe119", "안동시": "#4363d8",
    "구미시": "#f58231", "영주시": "#911eb4", "영천시": "#3cb44b", "상주시": "#f032e6",
    "문경시": "#bcf60c", "경산시": "#fabebe", "의성군": "#008080", "청송군": "#e6beff",
    "영양군": "#9a6324", "영덕군": "#000000", "청도군": "#800000", "고령군": "#aaffc3",
    "성주군": "#808000", "칠곡군": "#ffd8b1", "예천군": "#000075", "봉화군": "#808080",
    "울진군": "#4682B4", "울릉군": "#f7f2bd"
}

# ====== [5] 영천시 데이터 로딩 ======
try:
    # 영천시 저수지 데이터
    df_yeongcheon = pd.read_excel(DATA_DIR / '경상북도_영천시_저수지및댐.xlsx').dropna()
    
    # 반려동물 동반 가능 시설 위치 리스트
    locations = [
        (36.01841762, 128.929917), (35.97173253, 128.939907), (35.95973738, 128.93954),
        (35.96594607, 128.918217), (35.93153836, 128.87455), (35.91826818, 129.011153),
        (35.96426248, 128.924962), (35.93361167, 128.876258), (35.9719872, 128.941242),
        (35.96837891, 128.933538), (35.97188757, 128.939926), (35.99097629, 128.823406),
        (35.96468716, 128.938253), (35.96440713, 128.926174), (35.96371503, 128.939265),
        (35.93358914, 128.871295), (35.9721287, 128.93577), (35.93067846, 128.870576),
        (36.04169724, 128.787972), (35.96482654, 128.93923), (35.96359599, 128.936706),
        (35.97502046, 128.947498), (35.95795857, 128.913141), (35.98890075, 128.95512),
        (35.9581778, 128.913096), (35.96122769, 128.92841), (35.97523599, 128.94772),
        (36.04494524, 128.799646), (35.97216113, 128.937574), (36.03276243, 128.889247),
        (35.94410222, 128.897127), (36.01859865, 128.929978), (35.99026252, 128.794311),
        (36.12326392, 128.901235), (35.97438618, 128.945949), (35.98250081, 128.95221),
        (35.9722037, 128.935942), (35.95180542, 128.930731), (35.90275615, 128.856568),
        (35.9584334, 128.909988), (36.05347739, 128.89304), (35.97592922, 128.953099),
        (35.90275615, 128.856568),
    ]
    
    # 반경 2km 시설 수 계산
    def count_nearby_facilities(lat, lon, locations, radius_km=2):
        count = 0
        for loc_lat, loc_lon in locations:
            if haversine(lat, lon, loc_lat, loc_lon) <= radius_km:
                count += 1
        return count

    df_yeongcheon['반경2km_시설수'] = df_yeongcheon.apply(
        lambda row: count_nearby_facilities(row['위도'], row['경도'], locations), axis=1
    )

    # 정규화 함수
    def normalize(series): 
        return (series - series.min()) / (series.max() - series.min())

    df_yeongcheon['면적_정규화'] = normalize(df_yeongcheon['면적'])
    df_yeongcheon['둘레_정규화'] = normalize(df_yeongcheon['둘레'])
    df_yeongcheon['시설수_정규화'] = normalize(df_yeongcheon['반경2km_시설수'])

    # 중심지와의 거리 계산
    centers = [
        (35.92646737, 128.8823282), (36.01411727, 129.02104376),
        (35.96487293, 128.94139421), (36.05822495, 128.89287688),
        (36.03249647, 128.79147335)
    ]
    
    def get_closest_center_distance(lat, lon):
        return min(haversine(lat, lon, c_lat, c_lon) for c_lat, c_lon in centers)

    df_yeongcheon['중심거리_km'] = df_yeongcheon.apply(
        lambda row: get_closest_center_distance(row['위도'], row['경도']), axis=1
    )
    df_yeongcheon['거리_정규화'] = 1 - normalize(df_yeongcheon['중심거리_km'])

    # 적합도 점수 계산
    df_yeongcheon['적합도점수'] = (
        0.3 * df_yeongcheon['면적_정규화'] +
        0.3 * df_yeongcheon['둘레_정규화'] +
        0.2 * df_yeongcheon['거리_정규화'] +
        0.2 * df_yeongcheon['시설수_정규화']
    )

    # 행정동 결합
    gdf_yeongcheon = gpd.read_file(DATA_DIR / "BND_ADM_DONG_PG.shp", encoding='cp949')
    gdf_yeongcheon = gdf_yeongcheon.to_crs(epsg=4326)
    yc_codes = [
        "37070330", "37070340", "37070350", "37070360", "37070370", "37070380",
        "37070520", "37070540", "37070550", "37070510", "37070310", "37070320",
        "37070110", "37070390", "37070400", "37070530"
    ]
    gdf_yeongcheon = gdf_yeongcheon[gdf_yeongcheon["ADM_CD"].astype(str).isin(yc_codes)]

    df_yeongcheon['geometry'] = [Point(xy) for xy in zip(df_yeongcheon['경도'], df_yeongcheon['위도'])]
    df_yeongcheon_gdf = gpd.GeoDataFrame(df_yeongcheon, geometry='geometry', crs='EPSG:4326')
    df_yeongcheon = df_yeongcheon.join(
        gpd.sjoin(df_yeongcheon_gdf, gdf_yeongcheon[['geometry', 'ADM_CD', 'ADM_NM']], 
                 how='left', predicate='within')[['ADM_NM']], how='left'
    )
    df_yeongcheon.rename(columns={'ADM_NM': '행정동명'}, inplace=True)
    
    unique_areas = ["전체"] + sorted(df_yeongcheon['행정동명'].dropna().unique().tolist())
    
except Exception as e:
    print(f"영천시 데이터 로딩 오류: {e}")
    df_yeongcheon = pd.DataFrame()
    unique_areas = ["전체"]

# ====== [6] 경상북도 Shapefile 로딩 ======
def load_and_optimize_shapefile(shp_path: str):
    """Shapefile을 로드하고 성능을 위해 최적화"""
    encodings = ['cp949', 'euc-kr', 'utf-8', 'latin1']
    gdf = None
    
    for encoding in encodings:
        try:
            gdf = gpd.read_file(shp_path, encoding=encoding)
            break
        except Exception as e:
            continue
    
    if gdf is None:
        raise Exception("모든 인코딩 시도 실패")
    
    # CRS 확인 및 설정
    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    elif gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # 지오메트리 단순화
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001, preserve_topology=True)
    
    # 지역명 컬럼 찾기
    potential_name_cols = [
        'SIGUNGU_NM', 'SGG_NM', 'SIGUNGU', 'SGG', 'NAME', 'name',
        'adm_nm', 'ADM_NM', 'full_nm', 'FULL_NM', 'sig_nm', 'SIG_NM'
    ]
    
    name_col = None
    for col in potential_name_cols:
        if col in gdf.columns:
            name_col = col
            break
    
    if name_col is None:
        for col in gdf.columns:
            if gdf[col].dtype == 'object' and col != 'geometry':
                sample_values = gdf[col].dropna().head(3).tolist()
                if any(any(keyword in str(val) for keyword in ['시', '군', '구']) for val in sample_values):
                    name_col = col
                    break
    
    if name_col is None:
        raise Exception("지역명 컬럼을 찾을 수 없습니다")
    
    gdf = gdf[['geometry', name_col]].copy()
    gdf.rename(columns={name_col: '행정구역'}, inplace=True)
    gdf['행정구역'] = gdf['행정구역'].astype(str).str.strip()
    
    # 경상북도 지역만 필터링
    gyeongbuk_pattern = '|'.join(REGIONS)
    gdf = gdf[gdf['행정구역'].str.contains(gyeongbuk_pattern, na=False, regex=True)]
    gdf = gdf[~gdf.geometry.is_empty]
    
    # 포항시 통합
    gdf.loc[gdf['행정구역'].str.contains('포항시', na=False), '행정구역'] = '포항시'
    
    # 중복 지역 통합
    duplicate_regions = gdf['행정구역'].value_counts()
    if len(duplicate_regions[duplicate_regions > 1]) > 0:
        gdf_list = []
        for region in gdf['행정구역'].unique():
            region_gdf = gdf[gdf['행정구역'] == region]
            if len(region_gdf) > 1:
                unified_geometry = region_gdf.geometry.unary_union
                gdf_list.append({'행정구역': region, 'geometry': unified_geometry})
            else:
                gdf_list.append({'행정구역': region, 'geometry': region_gdf.geometry.iloc[0]})
        gdf_final = gpd.GeoDataFrame(gdf_list, crs='EPSG:4326')
    else:
        gdf_final = gdf.copy()
    
    return gdf_final

try:
    gdf_gyeongbuk = load_and_optimize_shapefile(DATA_DIR / "BND_SIGUNGU_PG.shp")
    unique_gyeongbuk_areas = sorted(gdf_gyeongbuk['행정구역'].unique())
except Exception as e:
    print(f"경상북도 shapefile 로딩 오류: {e}")
    gdf_gyeongbuk = gpd.GeoDataFrame()
    unique_gyeongbuk_areas = []

# ====== [7] 경상북도 분석 함수들 ======
def analyze_air_pollution_data(file_path: str) -> pd.DataFrame:
    """대기오염 데이터 분석"""
    pollutants = {
        'PM2.5': '미세먼지_PM2.5__월별_도시별_대기오염도',
        'PM10': '미세먼지_PM10__월별_도시별_대기오염도',
        'O3': '오존_월별_도시별_대기오염도',
        'CO': '일산화탄소_월별_도시별_대기오염도',
        'NO2': '이산화질소_월별_도시별_대기오염도'
    }
    
    result_df = None
    for pollutant, sheet_name in pollutants.items():
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            gyeongbuk_df = df[df['구분(1)'] == '경상북도']
            month_cols = [col for col in df.columns if str(col).replace('.', '').isdigit()]
            avg_df = gyeongbuk_df.groupby('구분(2)')[month_cols].mean().mean(axis=1).reset_index()
            avg_df.columns = ['시군구', f'{pollutant}_평균']
            
            if result_df is None:
                result_df = avg_df
            else:
                result_df = pd.merge(result_df, avg_df, on='시군구', how='outer')
        except Exception as e:
            pass
    
    return result_df

def analyze_crime_rate(crime_file_path, population_file_path):
    """범죄율 데이터 분석"""
    crime_df = pd.read_excel(crime_file_path)
    region_columns = [col for col in crime_df.columns if col not in ['범죄대분류', '범죄중분류']]
    total_crimes = crime_df[region_columns].sum().reset_index()
    total_crimes.columns = ['시군구', '총범죄건수']

    pop_raw = pd.read_excel(population_file_path, sheet_name="1-2. 읍면동별 인구 및 세대현황", header=[3,4])
    pop_df = pop_raw[[("구분","Unnamed: 0_level_1"),("총계","총   계")]].copy()
    pop_df.columns = ["region", "population"]
    pop_df = unify_and_filter_region(pop_df, "region")

    crime_data = total_crimes.copy()
    crime_data['region'] = crime_data['시군구']

    merged = pd.merge(crime_data[['region', '총범죄건수']], pop_df, on="region", how="inner")
    merged["범죄율"] = merged["총범죄건수"] / merged["population"]
    merged = merged.sort_values("범죄율", ascending=False)
    return merged

def analyze_accident_data(excel_path: str) -> pd.DataFrame:
    """교통사고 데이터 분석"""
    df = pd.read_excel(excel_path)
    df = df.loc[df['구분'] == '사고']
    df = df.drop(columns=['연도', '구분']).mean()

    mapping_dict = {
        '포항북부': '포항시', '포항남부': '포항시', '경주': '경주시', '김천': '김천시',
        '안동': '안동시', '구미': '구미시', '영주': '영주시', '영천': '영천시', '상주': '상주시',
        '문경': '문경시', '경산': '경산시', '의성': '의성군', '청송': '청송군', '영양': '영양군',
        '영덕': '영덕군', '청도': '청도군', '고령': '고령군', '성주': '성주군', '칠곡': '칠곡군',
        '예천': '예천군', '봉화': '봉화군', '울진': '울진군', '울릉': '울릉군'
    }

    city_accident_avg = defaultdict(list)
    for region, value in df.items():
        std_city = mapping_dict.get(region)
        if std_city:
            city_accident_avg[std_city].append(value)

    city_accident_avg = {city: sum(values) / len(values) for city, values in city_accident_avg.items()}
    acc_df = pd.DataFrame({'시군': list(city_accident_avg.keys()), '평균사고건수': list(city_accident_avg.values())})
    pop_df_local = pd.DataFrame({'시군': REGIONS, '인구수': POPULATION})
    merged_df = pd.merge(acc_df, pop_df_local, on='시군')
    merged_df['사고비율'] = merged_df['평균사고건수'] / merged_df['인구수']
    merged_df = merged_df.sort_values("사고비율", ascending=False)
    return merged_df

def analyze_park_area(excel_path: str) -> pd.DataFrame:
    """공원면적 데이터 분석"""
    df = pd.read_excel(excel_path)
    df_subset = df.iloc[3:, [1, 3]]
    df_subset.columns = ['시군', '면적']
    merged_df = pd.merge(df_subset, pop_df, on='시군')
    merged_df['면적'] = pd.to_numeric(merged_df['면적'], errors='coerce')
    merged_df['인구수'] = pd.to_numeric(merged_df['인구수'], errors='coerce')
    merged_df['공원면적비율'] = merged_df['면적'] / merged_df['인구수']
    merged_df = merged_df.sort_values("공원면적비율", ascending=True)
    return merged_df

def analyze_population_facility_ratio(facility_file_path: str, population_file_path: str) -> pd.DataFrame:
    """반려동물 시설 데이터 분석"""
    facility_df = pd.read_csv(facility_file_path, encoding="cp949")
    facility_df = unify_and_filter_region(facility_df, "시도 명칭", "시군구 명칭")

    pop_raw = pd.read_excel(population_file_path, sheet_name="1-2. 읍면동별 인구 및 세대현황", header=[3, 4])
    pop_df_local = pop_raw[[("구분", "Unnamed: 0_level_1"), ("총계", "총   계")]].copy()
    pop_df_local.columns = ["region", "population"]
    pop_df_local = unify_and_filter_region(pop_df_local, "region")
    pop_df_local = pop_df_local[pop_df_local["region"].isin(facility_df["region"].unique())]
    
    df_fac_cnt = facility_df.groupby("region").size().reset_index(name="facility_count")
    df_merge = pd.merge(df_fac_cnt, pop_df_local, on="region")
    df_merge["per_person"] = df_merge["facility_count"] / df_merge["population"]
    df_merge = df_merge.sort_values("per_person", ascending=True)
    return df_merge

# ====== [8] 레이더 차트 함수 ======
def plot_radar_chart(park_fp, acc_fp, facility_fp, pop_fp, crime_fp, pollution_fp, selected_regions=None):
    """종합 레이더 차트"""
    # 데이터 준비 및 분석 (기존 코드와 동일)
    df_park = pd.read_excel(park_fp).iloc[3:,[1,3]]
    df_park.columns = ['시군','면적']
    df_park['면적'] = pd.to_numeric(df_park['면적'],errors='coerce')
    df_park = df_park.merge(pop_df, on='시군')
    df_park['per_person'] = df_park['면적'] / df_park['인구수']
    df_park['park_norm'] = df_park['per_person'] / df_park['per_person'].max()

    # 교통사고 분석
    df_acc = pd.read_excel(acc_fp)
    df_acc = df_acc[df_acc['구분'] == '사고'].drop(columns=['연도','구분'])
    acc_mean = df_acc.mean()
    mapping_acc = {
        '포항북부':'포항시','포항남부':'포항시','경주':'경주시','김천':'김천시','안동':'안동시','구미':'구미시',
        '영주':'영주시','영천':'영천시','상주':'상주시','문경':'문경시','경산':'경산시','의성':'의성군',
        '청송':'청송군','영양':'영양군','영덕':'영덕군','청도':'청도군','고령':'고령군','성주':'성주군',
        '칠곡':'칠곡군','예천':'예천군','봉화':'봉화군','울진':'울진군','울릉':'울릉군'
    }
    acc_dict = {}
    for k, v in acc_mean.items():
        city = mapping_acc.get(k)
        if city:
            acc_dict.setdefault(city, []).append(v)
    acc_avg = {city: np.mean(vals) for city, vals in acc_dict.items()}
    df_acc2 = pd.DataFrame.from_dict(acc_avg, orient='index', columns=['acc']).reset_index().rename(columns={'index': '시군'})
    df_acc2 = df_acc2.merge(pop_df, on='시군')
    df_acc2['acc_inv'] = 1 / (df_acc2['acc'] / df_acc2['인구수'])
    df_acc2['acc_norm'] = df_acc2['acc_inv'] / df_acc2['acc_inv'].max()

    # 반려동물 시설 분석
    df_fac = pd.read_csv(facility_fp, encoding='cp949')
    df_fac = df_fac[df_fac['시도 명칭'] == '경상북도']
    df_fac['시군'] = df_fac['시군구 명칭'].str.extract(r'^(.*?[시군])')[0]
    df_fac = df_fac[df_fac['시군'] != '군위군']
    fac_counts = df_fac['시군'].value_counts().rename_axis('시군').reset_index(name='facility_count')
    fac_df = fac_counts.merge(pop_df, on='시군')
    fac_df['per_person'] = fac_df['facility_count'] / fac_df['인구수']
    fac_df['fac_norm'] = fac_df['per_person'] / fac_df['per_person'].max()

    # 범죄 분석
    crime_df = pd.read_excel(crime_fp)
    cols = [c for c in crime_df.columns if c not in ['범죄대분류', '범죄중분류']]
    crime_tot = crime_df[cols].sum().reset_index()
    crime_tot.columns = ['raw', 'crime']
    crime_tot['시군'] = crime_tot['raw'].str.split().str[0]
    crime_tot = crime_tot[crime_tot['시군'] != '군위군'][['시군', 'crime']]
    crime_tot = crime_tot.merge(pop_df, on='시군')
    crime_tot['crime_inv'] = 1 / (crime_tot['crime'] / crime_tot['인구수'])
    crime_tot['crime_norm'] = crime_tot['crime_inv'] / crime_tot['crime_inv'].max()

    # 대기오염 분석
    pollutants = {
        'PM2.5': '미세먼지_PM2.5__월별_도시별_대기오염도',
        'PM10': '미세먼지_PM10__월별_도시별_대기오염도',
        'O3': '오존_월별_도시별_대기오염도',
        'CO': '일산화탄소_월별_도시별_대기오염도',
        'NO2': '이산화질소_월별_도시별_대기오염도'
    }
    polls = []
    for pol, sheet in pollutants.items():
        try:
            dfp = pd.read_excel(pollution_fp, sheet_name=sheet)
            dfp = dfp[dfp['구분(1)'] == '경상북도']
            mcols = [c for c in dfp.columns if c not in ['구분(1)', '구분(2)']]
            dfp[mcols] = dfp[mcols].apply(pd.to_numeric, errors='coerce')
            avg = dfp.groupby('구분(2)')[mcols].mean().mean(axis=1).rename(pol)
            polls.append(avg)
        except:
            pass
    
    if polls:
        poll_df = pd.concat(polls, axis=1).reset_index().rename(columns={'구분(2)': '시군'})
        poll_df['시군'] = poll_df['시군'].astype(str).apply(lambda x: x + '시' if not x.endswith(('시', '군')) else x)
        for pol in pollutants:
            if pol in poll_df.columns:
                poll_df[f'{pol}_n'] = poll_df[pol] / poll_df[pol].max()
        poll_cols = [f'{p}_n' for p in pollutants if f'{p}_n' in poll_df.columns]
        poll_df['poll_comp'] = poll_df[poll_cols].sum(axis=1)
        poll_df['poll_inv'] = 1 / poll_df['poll_comp']
        poll_df['poll_norm'] = poll_df['poll_inv'] / poll_df['poll_inv'].max()
    else:
        poll_df = pd.DataFrame({'시군': REGIONS, 'poll_norm': [0.5] * len(REGIONS)})

    # 통합 및 시각화
    metrics = pd.DataFrame({'시군': REGIONS})
    metrics = metrics.merge(df_park[['시군', 'park_norm']], on='시군', how='left')
    metrics = metrics.merge(df_acc2[['시군', 'acc_norm']], on='시군', how='left')
    metrics = metrics.merge(fac_df[['시군', 'fac_norm']], on='시군', how='left')
    metrics = metrics.merge(crime_tot[['시군', 'crime_norm']], on='시군', how='left')
    metrics = metrics.merge(poll_df[['시군', 'poll_norm']], on='시군', how='left')
    
    # 결측값 처리
    metrics = metrics.fillna(0.5)

    categories = ['산책 환경', '반려동물 시설', '교통 안전', '치안', '대기 환경']
    theta = categories + [categories[0]]

    fig = go.Figure()
    
    for _, row in metrics.iterrows():
        values = [
            row['park_norm'], row['fac_norm'], row['acc_norm'],
            row['crime_norm'], row['poll_norm']
        ] + [row['park_norm']]

        is_selected = selected_regions and row['시군'] in selected_regions
        region_color = REGION_COLORS.get(row['시군'], '#808080')
        
        if is_selected:
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            rgb = hex_to_rgb(region_color)
            fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)'
            
            fig.add_trace(go.Scatterpolar(
                r=values, theta=theta, name=row['시군'],
                line=dict(width=3, color=region_color),
                opacity=1.0, fill='toself', fillcolor=fill_color
            ))
        else:
            fig.add_trace(go.Scatterpolar(
                r=values, theta=theta, name=row['시군'],
                line=dict(width=1, color='lightgray'),
                opacity=0.2, showlegend=False
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], side="clockwise", angle=90),
            angularaxis=dict(rotation=90, direction="clockwise")
        ),
        showlegend=True,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        width=520, height=500,
        margin=dict(t=20, b=0, l=0, r=0)
    )

    return fig

# ====== [9] 지도 생성 함수들 ======
def create_gyeongbuk_map(selected_regions=None):
    """경상북도 지도 생성"""
    if gdf_gyeongbuk.empty:
        return "<div>지도 데이터를 불러올 수 없습니다.</div>"
    
    bounds = gdf_gyeongbuk.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
        prefer_canvas=True,
        zoom_control=False
    )

    def style_function(feature):
        name = feature["properties"]["행정구역"]
        if selected_regions and name in selected_regions:
            col = REGION_COLORS.get(name, "#FF0000")
            return {
                "fillColor": col, "color": "#2e7d32", "weight": 3,
                "fillOpacity": 0.5, "opacity": 1.0
            }
        else:
            return {
                "fillColor": "#f5f5f5", "color": "#4caf50", "weight": 2,
                "fillOpacity": 0.5, "opacity": 0.8
            }

    folium.GeoJson(
        gdf_gyeongbuk,
        name="경북 행정구역",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["행정구역"],
            aliases=["지역:"],
            sticky=True
        )
    ).add_to(m)

    for idx, row in gdf_gyeongbuk.iterrows():
        centroid = row.geometry.centroid
        region_name = row["행정구역"]
        is_selected = selected_regions and region_name in selected_regions
        text_color = "#ffffff" if is_selected else "#333333"
        font_weight = "bold" if is_selected else "normal"
        font_size = "14px" if is_selected else "12px"
        
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f"""<div style="
                    font-family: 'Noto Sans KR', sans-serif;
                    font-size: {font_size};
                    font-weight: {font_weight};
                    color: {text_color};
                    text-align: center;
                    white-space: nowrap;
                    pointer-events: none;
                ">{region_name}</div>""",
                icon_size=(100, 20),
                icon_anchor=(50, 10)
            )
        ).add_to(m)
    
    return m._repr_html_()

def create_yeongcheon_map(selected_marker=None, map_type="normal", locations=None, selected_area=None):
    """영천시 지도 생성"""
    if df_yeongcheon.empty:
        return "<div>지도 데이터를 불러올 수 없습니다.</div>"
    
    # 기본 중심점과 줌 레벨
    center_lat, center_lng = 35.961380, 128.927778
    zoom = 11

    if locations is None:
        locations = pd.DataFrame()

    # 1. 읍면동 선택에 따른 지도 중심/줌 결정 (우선순위 높음)
    if selected_area and selected_area != "전체" and not gdf_yeongcheon.empty:
        area_gdf = gdf_yeongcheon[gdf_yeongcheon['ADM_NM'] == selected_area]
        if not area_gdf.empty:
            bounds = area_gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lng = (bounds[0] + bounds[2]) / 2
            zoom = 13  # 읍면동 확대
            
    # 2. 저수지가 선택되었고 locations에 해당 저수지가 있을 때만 위치 변경 (우선순위 낮음)
    elif selected_marker and not locations.empty and selected_marker in locations['시설명'].values:
        selected = locations[locations['시설명'] == selected_marker]
        if not selected.empty:
            center_lat, center_lng = selected.iloc[0]['위도'], selected.iloc[0]['경도']
            zoom = 15

    m = folium.Map(
        location=[center_lat, center_lng], 
        zoom_start=zoom, 
        width="100%", 
        height="100%",
        max_zoom=18,
        zoom_control=False
    )
    
    if map_type == "satellite":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI", 
            name="위성 지도", 
            control=False,
            max_zoom=18
        ).add_to(m)
    else:
        folium.TileLayer(
            "OpenStreetMap", 
            name="일반 지도", 
            control=False,
            max_zoom=19
        ).add_to(m)

    # 현재 지도 상태 유지를 위한 JavaScript 추가
    preserve_state_script = f"""
    <script>
    var selectedReservoir = "{selected_marker if selected_marker else ""}";
    var isReservoirSelected = selectedReservoir !== "" && selectedReservoir !== "None";
    
    setTimeout(function() {{
        var mapElement = document.querySelector('.folium-map');
        if (mapElement) {{
            var leafletMap = window[mapElement.id];
            if (leafletMap) {{
                if (!isReservoirSelected && sessionStorage.getItem('mapZoom') && sessionStorage.getItem('mapCenter')) {{
                    var savedZoom = parseInt(sessionStorage.getItem('mapZoom'));
                    var savedCenter = JSON.parse(sessionStorage.getItem('mapCenter'));
                    leafletMap.setView([savedCenter.lat, savedCenter.lng], savedZoom);
                }} else if (isReservoirSelected) {{
                    setTimeout(function() {{
                        var currentZoom = leafletMap.getZoom();
                        var currentCenter = leafletMap.getCenter();
                        sessionStorage.setItem('mapZoom', currentZoom);
                        sessionStorage.setItem('mapCenter', JSON.stringify({{
                            lat: currentCenter.lat,
                            lng: currentCenter.lng
                        }}));
                    }}, 1000);
                }}
                
                leafletMap.on('zoomend moveend', function() {{
                    setTimeout(function() {{
                        var currentZoom = leafletMap.getZoom();
                        var currentCenter = leafletMap.getCenter();
                        sessionStorage.setItem('mapZoom', currentZoom);
                        sessionStorage.setItem('mapCenter', JSON.stringify({{
                            lat: currentCenter.lat,
                            lng: currentCenter.lng
                        }}));
                    }}, 100);
                }});
            }}
        }}
    }}, 500);
    </script>
    """

    # 행정구역 경계
    if not gdf_yeongcheon.empty:
        folium.GeoJson(
            gdf_yeongcheon,
            name="영천시 읍면동 경계",
            style_function=lambda x: {
                'fillColor': 'transparent', 
                'color': 'DarkGreen', 
                'weight': 2,
                'fillOpacity': 0,
                'opacity': 0.7
            },
        ).add_to(m)

    if not locations.empty:
        for _, row in locations.iterrows():
            color = 'red' if selected_marker == row['시설명'] else 'blue'
            folium.Marker(
                location=[row['위도'], row['경도']],
                tooltip=row['시설명'],
                icon=folium.Icon(color=color)
            ).add_to(m)

    # 모든 저수지 점 표시
    for _, row in df_yeongcheon.iterrows():
        if locations.empty or row['시설명'] not in locations['시설명'].values:
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=3, 
                color='#1E90FF', 
                fill=True, 
                fill_color='#1E90FF', 
                fill_opacity=0.6,
                tooltip=row['시설명']
            ).add_to(m)

    # JavaScript 추가
    m.get_root().html.add_child(folium.Element(preserve_state_script))

    return m._repr_html_()

def create_barplot(data):
    """영천시 적합도 바 차트"""
    df_sorted = data.sort_values(by='적합도점수', ascending=True)
    fig = px.bar(
        df_sorted,
        x='적합도점수', y='시설명',
        orientation='h', title=f'상위 {len(data)}개 저수지 적합도 점수',
        height=300, width=320
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1.2], showgrid=False,
                  tickvals=[0, 0.5, 1.0], ticktext=['0', '0.5', '1']),
        yaxis_title=None, 
        margin=dict(l=0, r=10, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10)
    )
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='embed', config={'displayModeBar': False}))

# ====== [10] CSS 스타일 ======
custom_css = """
/* 헤더 스타일 */
#header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #1e3a8a;
    color: white;
    padding: 10px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* 헤더 로고 이미지 */
#header img {
  height: 30px;
  margin-right: 15px;
}

.tab-container {
    display: flex;
    gap: 10px;
}

.tab-button {
    background-color: rgba(255,255,255,0.2);
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 8px 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
}

.tab-button:hover {
    background-color: rgba(255,255,255,0.3);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.tab-button.active {
    background-color: rgba(255,255,255,0.9);
    color: #667eea;
    font-weight: bold;
}

/* 경상북도용 사이드바 */
#gyeongbuk-sidebar {
    position: fixed;
    top: 60px;
    left: 0;
    width: 300px;
    height: calc(100vh - 60px);
    background-color: white;
    box-shadow: 2px 0 8px rgba(0,0,0,0.2);
    transition: transform 0.3s ease-in-out;
    z-index: 1000;
    padding: 20px;
    transform: translateX(0);
}

#gyeongbuk-sidebar.open {
    transform: translateX(0);
}

#gyeongbuk-toggle-button {
    position: fixed;
    top: 80px;
    left: 300px;
    transform: translateX(-5px);
    z-index: 9999;
    background-color: white;
    border: 1px solid #ccc;
    width: 30px;
    height: 30px;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 0 6px rgba(0,0,0,0.2);
    cursor: pointer;
    transition: left 0.3s ease-in-out;
}

/* 영천시용 사이드바 */
#yeongcheon-sidebar {
    position: fixed;
    top: 60px;
    left: 0;
    width: 300px;
    height: calc(100vh - 60px);
    background-color: rgba(255,255,255,0.95);
    padding: 20px;
    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    z-index: 10000;
    transition: transform 0.3s ease-in-out;
    transform: translateX(0);
}

#yeongcheon-toggle-button {
    position: fixed;
    top: 80px;
    left: 300px;
    transform: translateX(-5px);
    z-index: 10001;
    background-color: white;
    border: 1px solid #ccc;
    width: 30px;
    height: 30px;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 0 6px rgba(0,0,0,0.2);
    cursor: pointer;
    transition: left 0.3s ease-in-out;
}

.sidebar-section {
    margin-bottom: 20px;
    border-bottom: 1px solid #eee;
    padding-bottom: 15px;
}

.close-btn {
    position: absolute;
    top: 10px;
    right: 15px;
    border: none;
    background: none;
    font-size: 18px;
    cursor: pointer;
    color: #999;
}

.close-btn:hover {
    color: #333;
}

:root {
  --bslib-sidebar-main-bg: #f3f3f3;
}

body {
  font-family: 'Noto Sans KR', sans-serif;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden;
}

h4 {
  margin-top: 0;
  font-weight: bold;
}
"""

# ====== [11] UI 구성 ======
app_ui = ui.page_fluid(
    ui.tags.style(custom_css),
    
    # 헤더
    ui.div(
        ui.div(
            # 좌측: 로고 + 제목
            ui.div(
                tags.img(src="/logo1.png", height="28px", style="margin-right: 10px;"),
                ui.h3("영천시 반려동물 산책로 분석 대시보드", style="margin: 0; font-size: 18px; color: white;"),
                style="display: flex; align-items: center;"
            ),
            # 우측: 탭 버튼
            ui.div(
                ui.HTML("""
                    <div class="tab-container">
                        <button class="tab-button active" id="tab-gyeongbuk"
                                onclick="setActiveTab('경상북도')">경상북도</button>
                        <button class="tab-button" id="tab-yeongcheon"
                                onclick="setActiveTab('영천시')">영천시</button>
                    </div>
                """),
                ui.tags.script("""
                    function setActiveTab(tabName) {
                        Shiny.setInputValue('top_tab', tabName);
                        document.querySelectorAll('.tab-button').forEach(btn => {
                            btn.classList.remove('active');
                        });
                        if (tabName === '경상북도') {
                            document.getElementById('tab-gyeongbuk').classList.add('active');
                        } else {
                            document.getElementById('tab-yeongcheon').classList.add('active');
                        }
                    }
                    
                    // 초기화 및 엔터키/체크박스 토글 기능
                    document.addEventListener("DOMContentLoaded", function() {
                        // 기본값을 경상북도로 설정
                        setTimeout(function() {
                            setActiveTab('경상북도');
                        }, 100);
                        
                        // 경상북도 관련 변수들
                        let isFirstTimeGyeongbuk = true;
                        let isFirstDetailsTime = true;
                        
                        // 체크박스 변경 시뮬레이션 함수 (경상북도용)
                        function simulateCheckboxChangeGyeongbuk() {
                            const firstChecked = document.querySelector("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
                            if (firstChecked) {
                                firstChecked.click(); // 해제
                                setTimeout(function() {
                                    firstChecked.click(); // 다시 체크
                                    setTimeout(function() {
                                        const applyBtn = document.getElementById('gyeongbuk_apply_selection');
                                        if (applyBtn) applyBtn.click(); // 분석 버튼 클릭
                                    }, 100);
                                }, 100);
                            }
                        }
                        
                        // 경상북도 분석 버튼 클릭 이벤트 (조건부 렌더링으로 변경)
                        document.addEventListener('click', function(event) {
                            if (event.target && event.target.id === 'gyeongbuk_apply_selection') {
                                const checkedBoxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
                                if (checkedBoxes.length > 0) {
                                    // 최초 메인 창이 열릴 때만 체크박스 변경 시뮬레이션
                                    if (isFirstTimeGyeongbuk) {
                                        isFirstTimeGyeongbuk = false;
                                        setTimeout(function() {
                                            simulateCheckboxChangeGyeongbuk();
                                        }, 700);
                                    }
                                } else {
                                    alert('분석할 지역을 먼저 선택해주세요.');
                                }
                            }
                            
                            // 지표 상세 버튼 클릭 처리
                            if (event.target && event.target.textContent === '자세히 보기') {
                                const container = document.getElementById('gyeongbuk-details-container');
                                if (container && isFirstDetailsTime) {
                                    isFirstDetailsTime = false;
                                    setTimeout(function() {
                                        simulateCheckboxChangeGyeongbuk();
                                    }, 500);
                                }
                            }
                        });
                        
                        // 엔터키 이벤트 (전체)
                        document.addEventListener('keydown', function(event) {
                            if (event.key === 'Enter' || event.keyCode === 13) {
                                const currentTab = document.querySelector('.tab-button.active');
                                if (currentTab && currentTab.id === 'tab-gyeongbuk') {
                                    // 경상북도 탭에서 엔터키
                                    const sidebar = document.getElementById('gyeongbuk-sidebar');
                                    if (sidebar && sidebar.classList.contains('open')) {
                                        const checkedBoxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
                                        if (checkedBoxes.length > 0) {
                                            const applyBtn = document.getElementById('gyeongbuk_apply_selection');
                                            if (applyBtn) applyBtn.click();
                                        }
                                    }
                                } else if (currentTab && currentTab.id === 'tab-yeongcheon') {
                                    // 영천시 탭에서 엔터키
                                    const sidebar = document.getElementById('yeongcheon-sidebar');
                                    const activeElement = document.activeElement;
                                    if (sidebar && (sidebar.contains(activeElement) || activeElement.tagName === 'INPUT' || activeElement.tagName === 'SELECT')) {
                                        // 숫자 입력 필드의 경우 blur 이벤트를 강제로 발생시켜 값 업데이트
                                        if (activeElement.tagName === 'INPUT' && activeElement.type === 'number') {
                                            activeElement.blur();
                                            activeElement.focus();
                                        }
                                        
                                        // 약간의 지연 후 Shiny 신호 전송
                                        setTimeout(function() {
                                            Shiny.setInputValue('yeongcheon_enter_key_pressed', Math.random(), {priority: 'event'});
                                        }, 50);
                                        
                                        event.preventDefault();
                                    }
                                }
                            }
                        });
                    });
                """),
                style="margin-left: auto;"
            ),
            id="header"
        ),
        style="""
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 60px;
            z-index: 9999;
        """
    ),

    # 조건부 컨텐츠
    ui.panel_conditional("input.top_tab == '경상북도'", ui.output_ui("gyeongbuk_ui")),
    ui.panel_conditional("input.top_tab == '영천시'", ui.output_ui("yeongcheon_ui")),
)

# ====== [12] 서버 함수 ======
def server(input, output, session):
    # 탭 변경 감지
    @reactive.Effect
    @reactive.event(input.top_tab)
    def on_tab_change():
        print(f"선택된 탭: {input.top_tab()}")

    # 경상북도 UI
    @output
    @render.ui
    def gyeongbuk_ui():
        return ui.TagList(
            # 지도
            ui.output_ui(
                "gyeongbuk_map",
                style="""
                    position: fixed;
                    top: 60px;
                    left: 0;
                    width: 100vw;
                    height: calc(100vh - 60px);
                    z-index: -1;
                """
            ),

            # 사이드바 토글 버튼
            ui.tags.button(
                "〈",
                id="gyeongbuk-toggle-button",
                onclick="""
                    const sidebar = document.getElementById('gyeongbuk-sidebar');
                    const btn = document.getElementById('gyeongbuk-toggle-button');
                    const isClosed = sidebar.style.transform === 'translateX(-280px)';
                    sidebar.style.transform = isClosed ? 'translateX(0)' : 'translateX(-280px)';
                    sidebar.classList.toggle('open', !isClosed);
                    btn.innerText = isClosed ? '〈' : '〉';
                    btn.style.left = isClosed ? '300px' : '20px';
                """
            ),

            # 사이드바
            ui.div(
                ui.h3("경상북도 지역 선택", style="margin-bottom: 20px;"),
                ui.input_checkbox_group(
                    "gyeongbuk_selected_areas", 
                    "", 
                    choices={
                        area: ui.HTML(f"""
                            <span style="display: inline-flex; align-items: center;">
                                {area}
                                <span style="
                                    display: inline-block;
                                    width: 12px;
                                    height: 12px;
                                    background-color: {REGION_COLORS.get(area, '#808080')};
                                    border-radius: 50%;
                                    margin-left: 8px;
                                    border: 1px solid #ddd;
                                "></span>
                            </span>
                        """) for area in unique_gyeongbuk_areas
                    },
                    selected=[]
                ),
                
                ui.div(
                    ui.tags.button(
                        "모두선택",
                        onclick="""
                            const checkboxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']");
                            checkboxes.forEach(checkbox => {
                                if (!checkbox.checked) {
                                    checkbox.click();
                                }
                            });
                        """,
                        style="width: 44%; padding: 8px; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-size: 12px; font-weight: bold; cursor: pointer; margin-right: 4%;"
                    ),
                    ui.tags.button(
                        "모두해제",
                        onclick="""
                            const checkboxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']");
                            checkboxes.forEach(checkbox => {
                                if (checkbox.checked) {
                                    checkbox.click();
                                }
                            });
                        """,
                        style="width: 44%; padding: 8px; background-color: #f44336; color: white; border: none; border-radius: 4px; font-size: 12px; font-weight: bold; cursor: pointer;"
                    ),
                    style="margin-top: 10px; margin-bottom: 15px; display: flex;"
                ),
                
                ui.input_action_button("gyeongbuk_apply_selection", "선택 지역 분석하기", 
                                    style="width: 92%; margin-top: 10px; padding: 10px; background-color: #4caf50; color: white; border: none; border-radius: 5px; font-weight: bold;"),
                id="gyeongbuk-sidebar",
                class_="open"
            ),

            # 팝업 창들
            ui.output_ui("gyeongbuk_popup"),
            ui.output_ui("gyeongbuk_details")
        )

    # 영천시 UI
    @output
    @render.ui  
    def yeongcheon_ui():
        return ui.TagList(
            # 지도
            ui.output_ui(
                "yeongcheon_map",
                style="""
                    position: fixed;
                    top: 60px;
                    left: 0;
                    width: 100vw;
                    height: calc(100vh - 60px);
                    z-index: -1;
                """
            ),

            # 사이드바 토글 버튼
            ui.tags.button(
                "〈",
                id="yeongcheon-toggle-button",
                onclick="""
                    const sidebar = document.getElementById('yeongcheon-sidebar');
                    const btn = document.getElementById('yeongcheon-toggle-button');
                    const isClosed = sidebar.style.transform === 'translateX(-280px)';
                    sidebar.style.transform = isClosed ? 'translateX(0)' : 'translateX(-280px)';
                    btn.innerText = isClosed ? '〈' : '〉';
                    btn.style.left = isClosed ? '300px' : '20px';
                """
            ),

            # 사이드바
            ui.div(
                ui.h3("영천시 저수지 분석", style="margin-bottom: 20px;"),
                ui.div(
                    ui.h4("저수지 필터링"),
                    ui.input_select("yeongcheon_area", "읍면동 선택", choices=unique_areas, selected="전체"),
                    ui.input_numeric("yeongcheon_top_n", "상위 저수지 개수", value=10, min=1, max=len(df_yeongcheon) if not df_yeongcheon.empty else 10),
                    class_="sidebar-section"
                ),
                ui.div(
                    ui.h4("가중치 설정"),
                    ui.input_slider("yeongcheon_weight_area", "면적 가중치", min=0, max=1, value=0.3, step=0.05),
                    ui.input_slider("yeongcheon_weight_perimeter", "둘레 가중치", min=0, max=1, value=0.3, step=0.05),
                    ui.input_slider("yeongcheon_weight_distance", "거리 가중치", min=0, max=1, value=0.2, step=0.05),
                    ui.input_slider("yeongcheon_weight_facilities", "시설수 가중치", min=0, max=1, value=0.2, step=0.05),
                    ui.input_action_button("yeongcheon_apply_filters", "입력", class_="btn btn-primary"),
                    class_="sidebar-section"
                ),
                id="yeongcheon-sidebar"
            ),

            # 지도 타입 선택
            ui.div(
                ui.input_radio_buttons(
                    "yeongcheon_map_type", "지도 종류 선택",
                    choices={"normal": "일반 지도", "satellite": "위성 지도"},
                    selected="normal", inline=True
                ),
                style="position: fixed; bottom: 20px; right: 20px; z-index: 9999; background-color: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); max-width: 300px;"
            ),

            # 동적 리스트와 차트
            ui.output_ui("yeongcheon_dynamic_list"),
            ui.output_ui("yeongcheon_dynamic_chart"),

            # 설명 박스
            ui.div(
                ui.output_ui("yeongcheon_description"),
                style="""
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    z-index: 9999;
                    background-color: rgba(255,255,255,0.95);
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                    max-width: 350px;
                """
            )
        )

    # ===== 경상북도 관련 서버 함수들 =====
    selected_regions_for_analysis = reactive.Value([])
    
    @reactive.effect
    @reactive.event(input.gyeongbuk_apply_selection)
    def update_gyeongbuk_analysis_regions():
        selected_regions_for_analysis.set(input.gyeongbuk_selected_areas() or [])
    
    @output()
    @render.ui
    def gyeongbuk_map():
        return ui.HTML(create_gyeongbuk_map(selected_regions=selected_regions_for_analysis()))

    @output()
    @render.ui
    def gyeongbuk_popup():
        sel = selected_regions_for_analysis()
        if not sel:
            return ui.div(style="display: none;")  # 빈 div 반환
        
        try:
            fig = plot_radar_chart(
                park_fp=DATA_DIR / "시군별_공원_면적.xlsx",
                acc_fp=DATA_DIR / "경상북도 시도별 교통사고 건수.xlsx",
                facility_fp=DATA_DIR / "한국문화정보원_전국 반려동물 동반 가능 문화시설 위치 데이터_20221130.csv",
                pop_fp=DATA_DIR / "경상북도 주민등록.xlsx",
                crime_fp=DATA_DIR / "경찰청_범죄 발생 지역별 통계.xlsx",
                pollution_fp=DATA_DIR / "월별_도시별_대기오염도.xlsx",
                selected_regions=sel
            )
            
            fig.update_layout(
                polar=dict(
                    domain=dict(x=[0.15, 0.75], y=[0.1, 0.9]),
                    radialaxis=dict(visible=True, range=[0, 1], side="clockwise", angle=90),
                    angularaxis=dict(rotation=90, direction="clockwise")
                ),
                showlegend=True,
                legend=dict(
                    orientation='v',
                    x=1.05,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle'
                ),
                width=520,
                height=330,
                margin=dict(t=30, b=20, l=20, r=140)
            )
            
            radar_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={'displayModeBar': False})
        except Exception as e:
            radar_html = f"<div>레이더 차트 생성 오류: {e}</div>"
        
        return ui.div(
            ui.tags.button(
                "×", 
                onclick="document.getElementById('gyeongbuk-popup-container').style.display = 'none';",
                style="""
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    border: none;
                    background: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                """
            ),
            ui.div(
                ui.h4("선택 지역 비교", style="margin: 0; display: inline-block;"),
                ui.tags.button(
                    "자세히 보기", 
                    onclick="""
                        const container = document.getElementById('gyeongbuk-details-container');
                        const btn = this;
                        if (container.style.display === 'none' || container.style.display === '') {
                            container.style.display = 'block';
                            btn.innerText = '닫기';
                            btn.style.backgroundColor = '#f44336';
                        } else {
                            container.style.display = 'none';
                            btn.innerText = '자세히 보기';
                            btn.style.backgroundColor = '#2196F3';
                        }
                    """,
                    style="""
                        margin-left: 10px;
                        padding: 4px 8px;
                        width: 70px;
                        background-color: #2196F3;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: bold;
                        cursor: pointer;
                        text-align: center;
                    """
                ),
                style="margin-top: 20px; margin-bottom: 15px; display: flex; align-items: center;"
            ),
            ui.div(
                ui.HTML(radar_html),
                style="height: 380px; margin-bottom: 10px; overflow: hidden;"
            ),
            id="gyeongbuk-popup-container",
            style="""
                position: fixed; 
                top: 20px; 
                right: 20px; 
                width: 520px; 
                height: 420px; 
                background-color: white; 
                padding: 20px; 
                box-shadow: 0 2px 12px rgba(0,0,0,0.3); 
                z-index: 9999; 
                border-radius: 12px; 
                overflow: hidden;
                display: block;
            """
        )

    @output()
    @render.ui
    def gyeongbuk_details():
        sel = selected_regions_for_analysis()
        if not sel:
            return ui.div()
        
        # 각 탭별 차트 생성
        try:
            # 대기오염 차트
            pollution_df = analyze_air_pollution_data(DATA_DIR / "월별_도시별_대기오염도.xlsx")
            if pollution_df is not None and not pollution_df.empty:
                pollutant_cols = [c for c in pollution_df.columns if c.endswith('_평균')]
                norm = pollution_df.copy()
                for col in pollutant_cols:
                    norm[col] = norm[col] / norm[col].max()
                norm['total_pollution'] = norm[pollutant_cols].sum(axis=1)
                norm_sel = norm[norm['시군구'].isin(sel)].sort_values('total_pollution', ascending=False)
                
                pollution_fig = go.Figure()
                for _, row in norm_sel.iterrows():
                    region = row['시군구']
                    hover_parts = [f"<b>{region}</b><br>"]
                    hover_parts.append(f"총 오염도: {row['total_pollution']:.3f}<br>")
                    for col in pollutant_cols:
                        pollutant_name = col.split('_')[0]
                        hover_parts.append(f"{pollutant_name}: {row[col]:.3f}<br>")
                    
                    pollution_fig.add_trace(go.Bar(
                        x=[region], y=[row['total_pollution']], name=region,
                        marker_color=REGION_COLORS.get(region, '#808080'),
                        hovertemplate=''.join(hover_parts) + '<extra></extra>',
                        showlegend=True
                    ))
                
                pollution_fig.update_layout(
                    width=580, height=350, showlegend=True,
                    legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                    yaxis=dict(title='총 대기오염 지수'),
                    xaxis=dict(tickangle=-45, automargin=True),
                    margin=dict(t=30, b=60, l=40, r=120),
                    template='plotly_white'
                )
                pollution_html = pollution_fig.to_html(full_html=False, include_plotlyjs="cdn", config={'displayModeBar': False})
            else:
                pollution_html = "<div>대기오염 데이터를 불러올 수 없습니다.</div>"

            # 범죄율 차트
            crime_df = analyze_crime_rate(DATA_DIR / "경찰청_범죄 발생 지역별 통계.xlsx", DATA_DIR / "경상북도 주민등록.xlsx")
            crime_sel = crime_df[crime_df['region'].isin(sel)]
            crime_fig = px.bar(crime_sel, x='region', y='범죄율', color='region',
                              color_discrete_map=REGION_COLORS, labels={'범죄율':'1인당 범죄율','region':''})
            crime_fig.update_layout(width=580, height=350, showlegend=True,
                                   legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                   xaxis=dict(automargin=True, tickangle=-45),
                                   margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            crime_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 범죄율: %{y:.5f}<extra></extra>')
            crime_html = crime_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

            # 교통사고 차트  
            traffic_df = analyze_accident_data(DATA_DIR / "경상북도 시도별 교통사고 건수.xlsx")
            traffic_sel = traffic_df[traffic_df['시군'].isin(sel)]
            traffic_fig = px.bar(traffic_sel, x='시군', y='사고비율', color='시군',
                                color_discrete_map=REGION_COLORS, labels={'사고비율':'1인당 평균 사고','시군':''})
            traffic_fig.update_layout(width=580, height=350, showlegend=True,
                                     legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                     xaxis=dict(automargin=True, tickangle=-45),
                                     margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            traffic_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 사고 건수: %{y:.5f}<extra></extra>')
            traffic_html = traffic_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

            # 공원면적 차트
            park_df = analyze_park_area(DATA_DIR / "시군별_공원_면적.xlsx")
            park_sel = park_df[park_df['시군'].isin(sel)]
            park_fig = px.bar(park_sel, x='시군', y='공원면적비율', color='시군',
                             color_discrete_map=REGION_COLORS, labels={'공원면적비율':'1인당 공원면적','시군':''})
            park_fig.update_layout(width=580, height=350, showlegend=True,
                                  legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                  xaxis=dict(automargin=True, tickangle=-45),
                                  margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            park_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 공원면적: %{y:.2f}㎡<extra></extra>')
            park_html = park_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

            # 반려동물 시설 차트
            facility_df = analyze_population_facility_ratio(DATA_DIR / "한국문화정보원_전국 반려동물 동반 가능 문화시설 위치 데이터_20221130.csv", DATA_DIR / "경상북도 주민등록.xlsx")
            facility_sel = facility_df[facility_df['region'].isin(sel)]
            facility_fig = px.bar(facility_sel, x='region', y='per_person', color='region',
                                 color_discrete_map=REGION_COLORS, labels={'per_person':'1인당 시설 수','region':''})
            facility_fig.update_layout(width=580, height=350, showlegend=True,
                                      legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                      xaxis=dict(automargin=True, tickangle=-45),
                                      margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            facility_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 시설 수: %{y:.6f}<extra></extra>')
            facility_html = facility_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

        except Exception as e:
            pollution_html = crime_html = traffic_html = park_html = facility_html = f"<div>차트 생성 오류: {e}</div>"

        return ui.div(
            ui.tags.button(
                "×", 
                onclick="""
                    document.getElementById('gyeongbuk-details-container').style.display = 'none';
                """,
                style="""
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    border: none;
                    background: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                """
            ),
            ui.h4("지표 상세 분석", style="margin-top: 20px;"),
            ui.div(
                ui.navset_tab(
                    ui.nav_panel("대기 환경", ui.HTML(pollution_html)),
                    ui.nav_panel("치안", ui.HTML(crime_html)),
                    ui.nav_panel("교통", ui.HTML(traffic_html)),
                    ui.nav_panel("산책 환경", ui.HTML(park_html)),
                    ui.nav_panel("반려동물 시설", ui.HTML(facility_html)),
                    selected="대기 환경"
                ),
                style="transform: scale(0.85); transform-origin: top left; width: 118%; height: 118%;"
            ),
            id="gyeongbuk-details-container",
            style="position: fixed; top: 460px; right: 20px; width: 520px; height: 420px; overflow: hidden; background-color: white; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.3); z-index: 9998; border-radius: 12px; display: none;"
        )

    # ===== 영천시 관련 서버 함수들 =====
    yeongcheon_selected_marker = reactive.Value(None)
    yeongcheon_show_list = reactive.Value(False)
    yeongcheon_show_chart = reactive.Value(False)
    yeongcheon_current_data = reactive.Value(pd.DataFrame())
    yeongcheon_button_clicks = reactive.Value({})

    @reactive.Effect
    @reactive.event(input.yeongcheon_apply_filters)  
    def handle_yeongcheon_apply():
        if df_yeongcheon.empty:
            return
        
        # 먼저 저수지 선택 완전히 초기화
        yeongcheon_selected_marker.set(None)
        
        if input.yeongcheon_area() == "전체":
            filtered = df_yeongcheon.copy()
        else:
            filtered = df_yeongcheon[df_yeongcheon['행정동명'] == input.yeongcheon_area()].copy()
        
        # 가중치 적용
        w_area = input.yeongcheon_weight_area()
        w_perimeter = input.yeongcheon_weight_perimeter()
        w_distance = input.yeongcheon_weight_distance()
        w_facilities = input.yeongcheon_weight_facilities()
        
        total = w_area + w_perimeter + w_distance + w_facilities
        if total == 0:
            w_area, w_perimeter, w_distance, w_facilities = 0.3, 0.3, 0.2, 0.2
            total = 1
        else:
            w_area /= total
            w_perimeter /= total
            w_distance /= total
            w_facilities /= total

        filtered['적합도점수'] = (
            w_area * filtered['면적_정규화'] +
            w_perimeter * filtered['둘레_정규화'] +
            w_distance * filtered['거리_정규화'] +
            w_facilities * filtered['시설수_정규화']
        )
        
        top_data = filtered.nlargest(max(1, input.yeongcheon_top_n()), '적합도점수').reset_index(drop=True)
        
        yeongcheon_current_data.set(top_data)
        yeongcheon_show_list.set(True)
        yeongcheon_show_chart.set(True)
        yeongcheon_button_clicks.set({})
        print(f"분석 실행됨 - 저수지 선택 초기화됨")

    # 엔터키 눌림 처리 추가 (영천시)
    @reactive.Effect
    @reactive.event(input.yeongcheon_enter_key_pressed)
    def handle_yeongcheon_enter_key():
        print("영천시 엔터키 감지됨 - 입력 버튼과 동일한 동작 실행")
        if df_yeongcheon.empty:
            return
        
        # 먼저 저수지 선택 완전히 초기화
        yeongcheon_selected_marker.set(None)
            
        if input.yeongcheon_area() == "전체":
            filtered = df_yeongcheon.copy()
        else:
            filtered = df_yeongcheon[df_yeongcheon['행정동명'] == input.yeongcheon_area()].copy()
        
        # 가중치 적용
        w_area = input.yeongcheon_weight_area()
        w_perimeter = input.yeongcheon_weight_perimeter()
        w_distance = input.yeongcheon_weight_distance()
        w_facilities = input.yeongcheon_weight_facilities()
        
        total = w_area + w_perimeter + w_distance + w_facilities
        if total == 0:
            w_area, w_perimeter, w_distance, w_facilities = 0.3, 0.3, 0.2, 0.2
            total = 1
        else:
            w_area /= total
            w_perimeter /= total
            w_distance /= total
            w_facilities /= total

        filtered['적합도점수'] = (
            w_area * filtered['면적_정규화'] +
            w_perimeter * filtered['둘레_정규화'] +
            w_distance * filtered['거리_정규화'] +
            w_facilities * filtered['시설수_정규화']
        )
        
        top_data = filtered.nlargest(max(1, input.yeongcheon_top_n()), '적합도점수').reset_index(drop=True)
        
        yeongcheon_current_data.set(top_data)
        yeongcheon_show_list.set(True)
        yeongcheon_show_chart.set(True)
        yeongcheon_button_clicks.set({})
        print(f"엔터키로 영천시 입력 실행됨 - 데이터 길이: {len(top_data)}, 저수지 선택 초기화됨")

    @output
    @render.ui
    def yeongcheon_map():
        if yeongcheon_show_list.get() and not yeongcheon_current_data.get().empty:
            return ui.HTML(create_yeongcheon_map(
                yeongcheon_selected_marker.get(), 
                input.yeongcheon_map_type(), 
                yeongcheon_current_data.get(),
                input.yeongcheon_area()
            ))
        else:
            return ui.HTML(create_yeongcheon_map(
                map_type=input.yeongcheon_map_type(), 
                locations=pd.DataFrame(),
                selected_area=input.yeongcheon_area()
            ))

    @output
    @render.ui
    def yeongcheon_dynamic_list():
        if not yeongcheon_show_list.get() or yeongcheon_current_data.get().empty:
            return ui.div()
        
        top_data = yeongcheon_current_data.get()
        
        return ui.div(
            ui.tags.button("✕", class_="close-btn", onclick="Shiny.setInputValue('yeongcheon_close_list', Math.random());"),
            ui.div(
                ui.h2(f"적합도 상위 {len(top_data)}개 저수지", style="font-size: 18px; margin-bottom: 15px;"),
                *[ui.input_action_button(f"yeongcheon_btn_{i}", 
                                       label=f"{i+1}. {row['시설명']}", 
                                       style="margin-bottom: 5px; width: 100%;")
                  for i, row in top_data.iterrows()],
                style="max-height: 350px; overflow-y: auto;"
            ),
            style="position: fixed; top: 80px; left: 320px; z-index: 9998; background-color: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); width: 320px;"
        )

    @output
    @render.ui
    def yeongcheon_dynamic_chart():
        if not yeongcheon_show_chart.get() or yeongcheon_current_data.get().empty:
            return ui.div()
        
        top_data = yeongcheon_current_data.get()
        
        return ui.div(
            ui.tags.button("✕", class_="close-btn", onclick="Shiny.setInputValue('yeongcheon_close_chart', Math.random());"),
            ui.div(
                create_barplot(top_data),
                style="margin-top: 25px;"
            ),
            style="position: fixed; top: 480px; left: 320px; z-index: 9998; background-color: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); width: 320px;"
        )

    @reactive.Effect
    @reactive.event(input.yeongcheon_close_list)
    def handle_yeongcheon_close_list():
        yeongcheon_show_list.set(False)

    @reactive.Effect
    @reactive.event(input.yeongcheon_close_chart)
    def handle_yeongcheon_close_chart():
        yeongcheon_show_chart.set(False)

    # 저수지 버튼 클릭 처리
    @reactive.Effect  
    def handle_yeongcheon_button_clicks():
        if not yeongcheon_show_list.get() or yeongcheon_current_data.get().empty:
            return
            
        top_data = yeongcheon_current_data.get()
        current_clicks = yeongcheon_button_clicks.get()
        
        for i, row in top_data.iterrows():
            btn_name = f"yeongcheon_btn_{i}"
            if hasattr(input, btn_name):
                current_value = input[btn_name]()
                if current_value and current_value > current_clicks.get(btn_name, 0):
                    yeongcheon_selected_marker.set(row['시설명'])
                    current_clicks[btn_name] = current_value
                    yeongcheon_button_clicks.set(current_clicks.copy())
                    print(f"저수지 선택됨: {row['시설명']}")
                    break

    @output
    @render.ui
    def yeongcheon_description():
        name = yeongcheon_selected_marker.get()
        
        if not name or df_yeongcheon.empty:
            return ui.p("저수지를 선택해 주세요.")
        
        row = df_yeongcheon[df_yeongcheon['시설명'] == name]
        if row.empty:
            return ui.p("정보를 찾을 수 없습니다.")
        
        row = row.iloc[0]
       
        return ui.div(
            ui.h3(name),
            ui.p(f"주소: {row['소재지지번주소']}"),
            ui.p(f"행정동: {row['행정동명']}"),
            ui.p(f"지도상 명칭: {row.get('지도상명칭', row.get('지도상 명칭', '정보 없음'))}"),
            ui.p(f"면적: {row['면적']} m²"),
            ui.p(f"둘레: {row['둘레']} m"),
            ui.p(f"인구 밀집지역과의 거리: {round(row['중심거리_km'], 2)} km"),
            ui.p(f"반려동물 동반 가능 시설 수 (2km 내): {row['반경2km_시설수']}개"),
            ui.p(f"적합도 점수: {round(row['적합도점수'], 3)}")
        )


here = os.path.dirname(__file__)
static_path = str(WWW_DIR)

app = App(app_ui, server, static_assets=static_path)