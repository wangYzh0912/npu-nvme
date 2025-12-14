#!/bin/bash

# NPU to NVMe Transfer Build Script

CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
CONFIG_FILE="${CURRENT_DIR}/.build_config"

# ==================================================
# Color Output
# ==================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ==================================================
# Load or Create Configuration
# ==================================================
load_or_create_config() {
    if [ -f "$CONFIG_FILE" ]; then
        print_info "Loading saved configuration..."
        source "$CONFIG_FILE"
        echo "  SPDK_ROOT_DIR: $SPDK_ROOT_DIR"
        echo "  SOC_VERSION: $SOC_VERSION"
        
        read -p "Use saved configuration? [Y/n]: " use_saved
        if [[ "$use_saved" =~ ^[Nn]$ ]]; then
            create_new_config
        fi
    else
        print_info "No saved configuration found"
        create_new_config
    fi
}

create_new_config() {
    echo ""
    echo "=========================================="
    echo "Creating Build Configuration"
    echo "=========================================="
    
    # SPDK路径
    local default_spdk="/home/user3/npu-nvme/spdk"
    read -p "Enter SPDK root directory [${default_spdk}]: " input_spdk
    SPDK_ROOT_DIR="${input_spdk:-$default_spdk}"
    
    # 验证SPDK路径
    while [ ! -d "$SPDK_ROOT_DIR" ]; do
        print_error "Directory does not exist: $SPDK_ROOT_DIR"
        read -p "Enter SPDK root directory: " SPDK_ROOT_DIR
    done
    
    # SOC版本
    local default_soc="Ascend310P3"
    echo ""
    echo "Available SOC versions: Ascend310P3, Ascend910B2, etc."
    read -p "Enter SOC version [${default_soc}]: " input_soc
    SOC_VERSION="${input_soc:-$default_soc}"
    
    # 保存配置
    echo ""
    read -p "Save this configuration? [Y/n]: " save_config
    if [[ !  "$save_config" =~ ^[Nn]$ ]]; then
        cat > "$CONFIG_FILE" << EOF
# Auto-generated build configuration
# You can edit this file or delete it to reconfigure

export SPDK_ROOT_DIR="$SPDK_ROOT_DIR"
export SOC_VERSION="$SOC_VERSION"
EOF
        print_info "Configuration saved to: $CONFIG_FILE"
    fi
    
    echo "=========================================="
}

# ==================================================
# Parse Arguments
# ==================================================
SHORT=v:,s:,c,d,r,h
LONG=soc-version:,spdk:,clean,debug,reconfigure,help
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")

if [ $?  != 0 ]; then
    print_error "Failed to parse arguments"
    exit 1
fi

eval set -- "$OPTS"

CLEAN=false
BUILD_TYPE="Release"
RECONFIGURE=false

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -s | --spdk)
        SPDK_ROOT_DIR="$2"
        shift 2
        ;;
    -c | --clean)
        CLEAN=true
        shift
        ;;
    -d | --debug)
        BUILD_TYPE="Debug"
        shift
        ;;
    -r | --reconfigure)
        RECONFIGURE=true
        shift
        ;;
    -h | --help)
        cat << EOF
Usage: $0 [OPTIONS]

Options:
    -v, --soc-version VERSION   SOC version
    -s, --spdk PATH            SPDK root directory
    -c, --clean                Clean build before compiling
    -d, --debug                Build in Debug mode
    -r, --reconfigure          Reconfigure build settings
    -h, --help                 Show this help

Examples:
    $0 --clean
    $0 --spdk /opt/spdk --clean
    $0 --reconfigure

EOF
        exit 0
        ;;
    --)
        shift
        break
        ;;
    *)
        print_error "Unexpected option: $1"
        exit 1
        ;;
    esac
done

# ==================================================
# Load Configuration
# ==================================================
if [ "$RECONFIGURE" = true ] || [ !  -f "$CONFIG_FILE" ]; then
    create_new_config
else
    load_or_create_config
fi

SOC_VERSION="${SOC_VERSION:-Ascend310P3}"

# ==================================================
# Setup Ascend Environment
# ==================================================
print_info "Setting up Ascend environment..."

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

if [ !  -f "$_ASCEND_INSTALL_PATH/bin/setenv.bash" ]; then
    print_error "Cannot find Ascend setenv.bash at: $_ASCEND_INSTALL_PATH"
    exit 1
fi

print_info "Sourcing: $_ASCEND_INSTALL_PATH/bin/setenv.bash"
source $_ASCEND_INSTALL_PATH/bin/setenv.bash

# 验证ACL头文件是否存在
if [ -f "$_ASCEND_INSTALL_PATH/include/acl/acl.h" ]; then
    print_info "✓ Found acl.h at: $_ASCEND_INSTALL_PATH/include/acl/acl.h"
else
    print_error "Cannot find acl.h in: $_ASCEND_INSTALL_PATH/include/acl/"
    print_info "Please check your CANN installation"
    exit 1
fi

# ==================================================
# Check SPDK
# ==================================================
print_info "Checking SPDK installation..."

if [ -z "$SPDK_ROOT_DIR" ]; then
    print_error "SPDK_ROOT_DIR is not set"
    exit 1
fi

if [ ! -d "$SPDK_ROOT_DIR" ]; then
    print_error "SPDK directory does not exist: $SPDK_ROOT_DIR"
    exit 1
fi

if [ !  -f "$SPDK_ROOT_DIR/build/lib/libspdk_nvme.a" ]; then
    print_error "SPDK not compiled.  Please compile SPDK first:"
    echo "  cd $SPDK_ROOT_DIR"
    echo "  ./configure"
    echo "  make -j\$(nproc)"
    exit 1
fi

print_info "✓ SPDK libraries found"

export SPDK_ROOT_DIR

# ==================================================
# Print Configuration
# ==================================================
echo ""
echo "=========================================="
echo "Build Configuration"
echo "=========================================="
echo "SOC Version:   ${SOC_VERSION}"
echo "Build Type:    ${BUILD_TYPE}"
echo "Ascend Path:   ${_ASCEND_INSTALL_PATH}"
echo "SPDK Path:     ${SPDK_ROOT_DIR}"
echo "Clean Build:   ${CLEAN}"
echo "=========================================="
echo ""

# ==================================================
# Build
# ==================================================
if [ "$CLEAN" = true ]; then
    print_info "Cleaning build directory..."
    rm -rf build out
fi

print_info "Configuring with CMake..."
mkdir -p build

# 使用详细输出来调试
cmake -B build \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_VERBOSE_MAKEFILE=ON

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

print_info "Building project..."
cmake --build build -j$(nproc) -- VERBOSE=1

if [ $?  -ne 0 ]; then
    print_error "Build failed"
    print_info "Check the error messages above"
    exit 1
fi

print_info "Installing..."
cmake --install build

if [ $? -ne 0 ]; then
    print_error "Installation failed"
    exit 1
fi

# ==================================================
# Success
# ==================================================
echo ""
echo "=========================================="
print_info "Build completed successfully!"
echo "Run script: ${CURRENT_DIR}/out/bin/run_test.sh"
echo "=========================================="