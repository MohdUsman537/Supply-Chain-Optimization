import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from collections import defaultdict
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
from geopy.distance import great_circle
from datetime import datetime
import os

# Constants
KM_PER_RADIAN = 6371
DEFAULT_EPSILON_KM = 300
MIN_SAMPLES = 3
OUTPUT_DIR = "SupplyChainReport"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define package_priority globally
package_priority = {"fragile": 1, "standard": 2, "perishable": 3}

def load_and_preprocess_data():
    """Load and preprocess the supply chain data."""
    df = pd.read_csv("supply_chain_data.csv")
    
    location_coordinates = {
        "Delhi": (28.7041, 77.1025), "Mumbai": (19.0760, 72.8777), 
        "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639), 
        "Bangalore": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567), "Jaipur": (26.9124, 75.7873), 
        "Lucknow": (26.8467, 80.9462), "Bhopal": (23.2599, 77.4126), 
        "Ahmedabad": (23.0225, 72.5714), "Chandigarh": (30.7333, 76.7794),
        "Indore": (22.7196, 75.8577), "Patna": (25.5941, 85.1376), 
        "Nagpur": (21.1458, 79.0882), "Surat": (21.1702, 72.8311), 
        "Visakhapatnam": (17.6868, 83.2185), "Ludhiana": (30.9010, 75.8573),
        "Vadodara": (22.3072, 73.1812), "Coimbatore": (11.0168, 76.9558)
    }
    
    df[['LATITUDE', 'LONGITUDE']] = df['LOCATION'].apply(
        lambda x: pd.Series(location_coordinates.get(x, (None, None))))
    
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    df['PACKAGE_PRIORITY'] = df['PACKAGE_TYPE'].map(package_priority)
    
    return df

def perform_clustering(df, epsilon_km=DEFAULT_EPSILON_KM):
    """Perform DBSCAN clustering on geographical coordinates."""
    coordinates = df[['LATITUDE', 'LONGITUDE']].values
    epsilon = epsilon_km / KM_PER_RADIAN
    
    db = DBSCAN(eps=epsilon, min_samples=MIN_SAMPLES, metric='haversine').fit(np.radians(coordinates))
    df['GroupNo'] = db.labels_
    
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = list(db.labels_).count(-1)
    
    print(f"ESTIMATED NUMBER OF CLUSTERS: {n_clusters}")
    print(f"ESTIMATED NUMBER OF NOISE POINTS: {n_noise}")
    
    return df, db.labels_

def generate_output_report(df):
    """Generate the priority-based output report."""
    output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for _, row in df.iterrows():
        output[row['DELIVERY_TIME']][(row['GroupNo'], row['LOCATION'])][row['PACKAGE_PRIORITY']].append(row['PRODUCT_ID'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(OUTPUT_DIR, f"delivery_priority_report_{timestamp}.txt")
    
    with open(report_filename, 'w') as f:
        for ind, (deliveryTime, geo_clusters) in enumerate(sorted(output.items()), start=1):
            f.write(f"PRIORITY - {ind} ({deliveryTime} DAYS) (GROUP):\n")
            
            for (geo_cluster, location), package_types in geo_clusters.items():
                f.write(f"    PRODUCTS FROM GROUP_NO-{geo_cluster} ({location}):\n")
                
                for package_priority_value, products in sorted(package_types.items()):
                    package_name = next(k for k, v in package_priority.items() if v == package_priority_value)
                    f.write(f"        PRIORITY-{package_priority_value}: {package_name} -> {products}\n")
            f.write("\n")
    
    print(f"REPORT SAVED TO {report_filename}")

def create_cluster_map(df):
    """Create an interactive map with clusters and delivery information."""
    colours = [
        'lightblue', 'green', 'lightred', 'lightgray', 'cadetblue', 'darkred', 
        'black', 'lightgreen', 'darkgreen', 'orange', 'gray', 'red', 'beige', 
        'purple', 'pink', 'darkpurple', 'darkblue', 'white', 'blue'
    ]
    
    cluster_ids = sorted(df['GroupNo'].unique())
    cluster_colors = {cluster_id: colours[i % len(colours)] for i, cluster_id in enumerate(cluster_ids)}
    
    map_center = [df['LATITUDE'].mean(), df['LONGITUDE'].mean()]
    m = folium.Map(location=map_center, zoom_start=5, tiles='cartodbpositron')
    
    cluster_delivery_times = {}
    cluster_sizes = df['GroupNo'].value_counts().to_dict()
    
    for cluster_id, group in df.groupby('GroupNo'):
        if cluster_id == -1:
            continue
            
        colour = cluster_colors[cluster_id]
        most_common_delivery_time = group['DELIVERY_TIME'].mode().iloc[0]
        cluster_delivery_times[cluster_id] = most_common_delivery_time
        
        for _, row in group.iterrows():
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=5,
                popup=folium.Popup(
                    f"<b>Cluster ID:</b> {row['GroupNo']}<br>"
                    f"<b>Location:</b> {row['LOCATION']}<br>"
                    f"<b>Product ID:</b> {row['PRODUCT_ID']}<br>"
                    f"<b>Package Type:</b> {row['PACKAGE_TYPE']}<br>"
                    f"<b>Delivery Time:</b> {most_common_delivery_time} Days<br>"
                    f"<b>Cluster Size:</b> {cluster_sizes.get(cluster_id, 0)}",
                    max_width=250
                ),
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.7
            ).add_to(m)
        
        unique_points = group[['LATITUDE', 'LONGITUDE']].drop_duplicates().values
        if len(unique_points) >= 3:
            hull = ConvexHull(unique_points)
            polygon = [(unique_points[vertex, 0], unique_points[vertex, 1]) for vertex in hull.vertices]
            centroid = np.mean(unique_points, axis=0)
            
            folium.Polygon(locations=polygon, color=colour, fill=True, fill_opacity=0.2).add_to(m)
            folium.Marker(
                location=centroid,
                icon=folium.DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(75,18),
                    html=f'<div style="font-size: 12pt; color: {colour}">Cluster {cluster_id}</div>'
                ),
                popup=f"Cluster {cluster_id} - Size: {cluster_sizes[cluster_id]}"
            ).add_to(m)
    
    legend_html = create_legend_html(cluster_colors, cluster_delivery_times, cluster_sizes)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    map_filename = os.path.join(OUTPUT_DIR, "supply_chain_clusters.html")
    m.save(map_filename)
    print(f"MAP SAVED TO {map_filename}")

def create_legend_html(cluster_colors, delivery_times, cluster_sizes):
    """Create HTML for the map legend."""
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; max-height: 300px;
        overflow-y: auto; width: auto; max-width: 300px; background-color: white;
        z-index: 9999; border: 2px solid grey; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); font-size: 14px;">
        <h4 style="margin-top: 0;">Cluster Information</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <th style="border-bottom: 1px solid #ddd; padding: 8px;">Cluster</th>
                <th style="border-bottom: 1px solid #ddd; padding: 8px;">Delivery</th>
                <th style="border-bottom: 1px solid #ddd; padding: 8px;">Size</th>
            </tr>
    """
    for cluster_id, color in sorted(cluster_colors.items()):
        if cluster_id == -1:
            continue
        legend_html += f"""
            <tr>
                <td style="border-bottom: 1px solid #ddd; padding: 8px;">
                    <span style="background-color: {color}; width: 16px; height: 16px;
                        display: inline-block; margin-right: 8px; border: 1px solid #333;"></span>
                    {cluster_id}
                </td>
                <td style="border-bottom: 1px solid #ddd; padding: 8px;">{delivery_times.get(cluster_id, "N/A")}</td>
                <td style="border-bottom: 1px solid #ddd; padding: 8px;">{cluster_sizes.get(cluster_id, 0)}</td>
            </tr>
        """
    legend_html += "</table><p><small>Click on markers for details</small></p></div>"
    return legend_html

def visualize_data(df):
    """Save visualizations of the supply chain data without displaying them."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='DELIVERY_TIME', hue='DELIVERY_TIME', legend=False)
    plt.title("Products per Delivery Time")
    plt.xlabel("Delivery Time (Days)")
    plt.ylabel("Product Count")
    
    plt.subplot(1, 2, 2)
    df['PACKAGE_TYPE'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
        colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
    plt.ylabel("")
    plt.title("Package Type Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "delivery_package_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df[df['GroupNo'] != -1], x="GroupNo", hue="PACKAGE_TYPE", 
        palette="Set2", hue_order=["fragile", "standard", "perishable"])
    plt.xlabel("Geographical Cluster")
    plt.ylabel("Number of Products")
    plt.title("Product Distribution by Location Clusters")
    plt.legend(title="Package Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_package_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[df['GroupNo'] != -1], x="GroupNo", y="DELIVERY_TIME")
    plt.xlabel("Geographical Cluster")
    plt.ylabel("Delivery Time (Days)")
    plt.title("Delivery Time Distribution by Cluster")
    plt.savefig(os.path.join(OUTPUT_DIR, "delivery_time_by_cluster.png"))
    plt.close()

def calculate_metrics(labels, coordinates):
    """Calculate clustering performance metrics."""
    core_points_mask = labels != -1
    core_labels = labels[core_points_mask]
    mask = labels != -1
    filtered_data = coordinates[mask]
    filtered_labels = labels[mask]
    print("ACCURACY OF THE MODEL: ")
    if len(set(core_labels)) > 1:
        coords_radians = np.radians(coordinates[core_points_mask])
        score = silhouette_score(coords_radians, core_labels, metric='haversine')
        print(f"\nSILHOUETTE SCORE (EXCLUDING NOISE): {score:.4f}")
    else:
        print("\nSILHOUETTE SCORE COULD NOT BE COMPUTED (ONLY ONE CLUSTER FOUND OR ALL NOISE).")
    if len(set(filtered_labels)) > 1:
        dbi = davies_bouldin_score(filtered_data, filtered_labels)
        print(f"DAVIES-BOULDIN INDEX: {dbi:.2f} (Lower is better)")
    else:
        print("DAVIES-BOULDIN INDEX: Not applicable (only 1 cluster after removing noise)")
    ch_score = calinski_harabasz_score(filtered_data, filtered_labels)
    print(f"CALINSKI-HARABASZ INDEX: {ch_score:.2f} (Higher is better)")    
    cluster_stats = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_points = coordinates[labels == cluster_id]
        if len(cluster_points) > 1:
            distances = [great_circle(p1, p2).km for i, p1 in enumerate(cluster_points) for p2 in cluster_points[i+1:]]
            cluster_stats[cluster_id] = {
                'avg_intra_distance_km': np.mean(distances),
                'max_intra_distance_km': np.max(distances),
                'size': len(cluster_points)
            }
    
    print("\nCLUSTER STATISTICS:")
    for cluster_id, stats in cluster_stats.items():
        print(f"CLUSTER {cluster_id}:")
        print(f"  SIZE: {stats['size']}")
        print(f"  AVERAGE INTRA-CLUSTER DISTANCE: {stats['avg_intra_distance_km']:.2f} KM")
        print(f"  MAXIMUM INTRA-CLUSTER DISTANCE: {stats['max_intra_distance_km']:.2f} KM")

def main():
    print("LOADING AND PREPROCESSING DATA...")
    df = load_and_preprocess_data()
    
    print("\nPERFORMING GEOGRAPHICAL CLUSTERING...")
    df, labels = perform_clustering(df)
    
    print("\nGENERATING OUTPUT REPORT...")
    generate_output_report(df)
    
    print("\nCREATING INTERACTIVE MAP...")
    create_cluster_map(df)
    
    print("\nSAVING VISUALIZATIONS...")
    visualize_data(df)
    
    print("\nCALCULATING CLUSTERING METRICS...")
    calculate_metrics(labels, df[['LATITUDE', 'LONGITUDE']].values)
    
    print("\nPROCESSING COMPLETE!")

if __name__ == "__main__":
    main()