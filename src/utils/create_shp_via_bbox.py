import shapefile
import os

# ===================================================================
#                      用户可自定义区域
# ===================================================================

# 1. 设置输出 Shapefile 的完整路径和文件名 (文件名末尾不要带 .shp)
# 注意: 请确保您对下面的目录有写入权限
output_dir = "/maps/zf281/btfm4rs/cci_workshop_roi_shp"
file_name = "external_request_#64"  # 在这里定义你想要的文件名

# 2. 定义ROI的四个角点坐标 (lon, lat)
# !!! 请在这里根据需要修改你的坐标 !!!

left_top = (115.7200, -20.6670)
right_top = (124.4100, -20.6670)
left_bottom = (115.7200, -24.3000)
right_bottom = (124.4100, -24.3000)


# ===================================================================
#                      脚本主逻辑 (通常无需修改)
# ===================================================================

print("脚本开始执行...")

# 确保输出目录存在，如果不存在则创建它
try:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"成功创建目录: {output_dir}")
except OSError as e:
    print(f"错误：无法创建目录 {output_dir}。请检查路径和权限。")
    print(f"系统错误信息: {e}")
    exit() # 如果目录无法创建，则退出脚本

# 组合成完整的文件路径
output_path = os.path.join(output_dir, file_name)

# 按照顺时针或逆时针顺序将角点组合成一个多边形的顶点列表
# Shapefile的多边形需要闭合，所以列表的第一个点和最后一个点必须相同
polygon_vertices = [
    left_top,
    right_top,
    right_bottom,
    left_bottom,
    left_top  # 回到起点以闭合多边形
]

try:
    # 使用 with 语句可以确保文件在完成后被正确保存和关闭
    with shapefile.Writer(output_path, shapefile.POLYGON) as w:
        # 添加一个属性字段，用于存储信息 (例如，一个ID)
        # 'C' 代表字符型, 'N' 代表数字型, 'D' 代表日期型
        w.field('ID', 'N')
        w.field('Name', 'C')

        # 添加几何图形 (我们定义的多边形)
        w.poly([polygon_vertices])

        # 为刚刚添加的几何图形，添加对应的属性记录
        w.record(1, 'ROI_Area')

    # 创建 .prj 文件来定义坐标系 (非常重要！)
    # 这里我们使用 WGS84 地理坐标系
    prj_content = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    with open(output_path + ".prj", "w") as prj_file:
        prj_file.write(prj_content)

    print("\n------------------------------------------------------")
    print("      Shapefile 已成功创建！")
    print("------------------------------------------------------")
    print(f"文件保存在以下位置: {output_path}.shp")
    print("一同生成的文件包括: .shx, .dbf, .prj")

except Exception as e:
    print(f"\n创建 Shapefile 时发生错误: {e}")
    print("请检查：")
    print("1. 是否已安装 'pyshp' 库 (pip install pyshp)。")
    print(f"2. 是否对目标目录 '{output_dir}' 有写入权限。")