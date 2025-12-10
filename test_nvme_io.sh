TEST_DIR=/home/user3/npu_nvme/nvme_test_dir
mkdir -p $TEST_DIR

# 运行fio顺序写入测试
# 参数说明：
# --directory: 测试目录
# --size: 测试数据总大小（10G，确保足够大以达到极限）
# --time_based --runtime=60s: 基于时间运行60秒
# --ramp_time=2s: 预热2秒
# --ioengine=libaio: 使用异步I/O
# --direct=1: 直接I/O，绕过缓存
# --verify=0: 无验证，纯速度测试
# --bs=1M: 块大小1M（适合顺序写入极限）
# --iodepth=64: 队列深度64（NVMe支持高并发）
# --rw=write: 纯写入
# --group_reporting=1: 汇总报告
# --numjobs=1: 单作业（可增加以测试多线程，但单线程常用于基线）
echo "开始测量NVMe极限写入速度..."
fio --name=nvme_write_limit \
    --directory=$TEST_DIR \
    --size=10G \
    --time_based \
    --runtime=60s \
    --ramp_time=2s \
    --ioengine=libaio \
    --direct=1 \
    --verify=0 \
    --bs=1M \
    --iodepth=64 \
    --rw=write \
    --group_reporting=1 \
    --numjobs=1

# 清理测试文件
echo "测试完成，清理临时文件..."
rm -rf $TEST_DIR