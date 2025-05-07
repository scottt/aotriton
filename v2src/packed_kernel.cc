// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/runtime.h>
#include <mutex>
#include <cstring>
#include <cassert>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <lzma.h>

#ifdef NDEBUG
#define AOTRITON_KERNEL_VERBOSE 0
#else
#define AOTRITON_KERNEL_VERBOSE 1
#endif

namespace fs = std::filesystem;
static const std::string_view KERNEL_STORAGE_V2_BASE = "aotriton.images";
static const std::string AKS2_MAGIC = "AKS2";
constexpr int AOTRITON_LZMA_BUFSIZ = 64 * 1024;

namespace {

const fs::path&
locate_aotriton_images() {
  static fs::path aotriton_images = []() {
    Dl_info info;
    dladdr((void*)locate_aotriton_images, &info);
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "dladdr locates libaotriton at: " << info.dli_fname << std::endl;
#endif
    return fs::path(info.dli_fname).parent_path() / KERNEL_STORAGE_V2_BASE;
  }();
  return aotriton_images;
}

}

#ifdef _MSC_VER

#include <io.h>
#if !defined(ssize_t)
    #include <BaseTsd.h> // For SSIZE_T
    typedef SSIZE_T ssize_t;
#endif

static int fd_open(const char *pathname, int msvc_flags) {
    return _open(pathname, msvc_flags | _O_BINARY);
}

static int fd_close(int fd) {
  return _close(fd);
}

static ssize_t fd_read(int fd, void *buf, size_t count) {
  unsigned int bytes_to_read;

  if (count == 0) {
    return 0;
  }

  // _read takes an unsigned int for the count.
  // Cap the count if it's larger than what unsigned int can hold.
  if (count > UINT_MAX) {
    bytes_to_read = UINT_MAX;
  } else {
    bytes_to_read = (unsigned int)count;
  }

  int bytes_actually_read = _read(fd, buf, bytes_to_read);
  return (ssize_t)bytes_actually_read; // _read returns int
}
#else // _MSC_VER
#include <unistd.h>
static int fd_open(const char *pathname, int msvc_flags) {
  return open(pathname, msvc_flags);
}

static int fd_close(int fd) {
  return close(fd);
}

static ssize_t fd_read(int fd, void *buf, size_t count) {
  return read(fd, buf, count);
}
#endif // _MSC_VER

namespace AOTRITON_NS {

std::shared_mutex PackedKernel::registry_mutex_;
std::unordered_map<std::string_view, PackedKernelPtr> PackedKernel::registry_;

PackedKernelPtr
PackedKernel::open(std::string_view package_path) {
  {
    // Fast path
    std::shared_lock lock(registry_mutex_);
    if (registry_.contains(package_path))
      return registry_[package_path];
  }

  // Slow path, registry doesn't contain this kernel
  std::unique_lock lock(registry_mutex_);
  // Prevent TOCTTOU b/w two locks
  if (registry_.contains(package_path))
    return registry_[package_path];
  const auto& storage_base = locate_aotriton_images();
  std::string rel_path(package_path);
  rel_path += ".aks2";
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "open " << rel_path << std::endl;
#endif
  const auto& u8_temp = (storage_base / rel_path).u8string();
  std::string p = std::string(reinterpret_cast<const char*>(u8_temp.data()),
                       u8_temp.size());
  int aks2fd = fd_open((const char *)(p.c_str()), O_RDONLY);
  if (aks2fd < 0) {
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "open(\"" << p << "\")"
              << " failed. errno: " << errno << std::endl;
#endif
    return nullptr;
  }
  auto ret = std::make_shared<PackedKernel>(aks2fd);
  fd_close(aks2fd);
  if (ret->status() == hipSuccess) {
    registry_.emplace(package_path, ret);
    return ret;
  }
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "PackedKernel::open(" << package_path << ") failed."
            << " Final status: " << hipGetErrorString(ret->status()) << std::endl;
#endif
  return nullptr;
}

struct AKS2_Header {
  char magic[4];
  uint32_t uncompressed_size;
  uint32_t number_of_kernels;
  uint32_t directory_size;
};

struct AKS2_Metadata {
  uint32_t shared_memory;
  uint32_t number_of_threads;
  uint32_t offset;
  uint32_t image_size;
  uint32_t filename_length;
};

// AKS2 Format
// -- Uncompressed
// 4B: AKS2  (AOTriton Kernel Storage version 2)
// 4B: Total uncompressed content size
// 4B: Number of Kernels (N)
// 4B: directory size
// -- Compressed
// N * varlen: (Directory)
//     4B shared memory size
//     4B number of threads in a GPU thread block
//     4B offset (from end of the header file)
//     4B image size
//     4B file name length (M), including trailing '\0'
//     MB file name
// N * varlen: Kernel Images (TODO: alignment requirements?)
PackedKernel::PackedKernel(int fd) {
  AKS2_Header header;
  auto header_read = fd_read(fd, &header, sizeof(AKS2_Header));
  if (header_read == sizeof(AKS2_MAGIC) && std::string_view(header.magic, 4) != AKS2_MAGIC) {
    final_status_ = hipErrorInvalidSource; // Broken at XZ level
    return;
  }
  decompressed_content_.resize(header.uncompressed_size);
  directory_.clear();

  lzma_stream strm = LZMA_STREAM_INIT;
  lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, 0);
  if (ret != LZMA_OK) {
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << " lzma_stream_decoder error: " << ret << std::endl;
#endif
    final_status_ = hipErrorInvalidSource; // Broken at XZ level
    return;
  }
  uint8_t inbuf[AOTRITON_LZMA_BUFSIZ];
  strm.next_in = nullptr;
  strm.avail_in = 0;
  strm.next_out = (uint8_t*)decompressed_content_.data();
  strm.avail_out = decompressed_content_.size();
  lzma_action action = LZMA_RUN;
  while (true) {
    if (strm.avail_in == 0) {
      strm.next_in = inbuf;
      auto rbytes = fd_read(fd, inbuf, AOTRITON_LZMA_BUFSIZ);
      if (rbytes <= 0) {
        action = LZMA_FINISH;
        break;
      }
      strm.avail_in = rbytes;
    }
    lzma_ret ret = lzma_code(&strm, action);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
      decompressed_content_.clear();
      directory_.clear();
      final_status_ = hipErrorIllegalState; // Content not fully decompressed
      return;
    }
  }
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "PackedKernel decompressed to " << (void*)decompressed_content_.data() << std::endl;
#endif
  const uint8_t* parse_ptr = decompressed_content_.data();
  for (uint32_t i = 0; i < header.number_of_kernels; i++) {
    auto metadata = reinterpret_cast<const AKS2_Metadata*>(parse_ptr);
    parse_ptr += sizeof(*metadata);
    std::string_view filename(reinterpret_cast<const char*>(parse_ptr));
    directory_.emplace(filename, metadata);
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "Add kernel " << i << ": " << filename << " offset: " << metadata->offset << std::endl;
#endif
    parse_ptr += metadata->filename_length;
  }
  kernel_start_ = parse_ptr;
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "PackedKernel.kernel_start_ = " << (void*)kernel_start_ << std::endl;
#endif
  if (kernel_start_ - decompressed_content_.data() != header.directory_size) {
    decompressed_content_.clear();
    directory_.clear();
    // Directory size not matching
    final_status_ = hipErrorIllegalAddress;
    return;
  }
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "PackedKernel.kernel_start_ sanity check passed" << std::endl;
#endif
  final_status_ = hipSuccess;
}

PackedKernel::~PackedKernel() {
}

TritonKernel::Essentials
PackedKernel::filter(std::string_view stem_name) const {
  if (status() != hipSuccess) {
    return { nullptr, 0, 0, dim3 { 0, 0, 0 } };
  }
  auto iter = directory_.find(stem_name);
  if (iter == directory_.end())
    return { nullptr, 0, 0, dim3 { 0, 1, 1 } };
  auto meta = iter->second;
  if (meta->image_size == 0) {
    // TODO: Sanity check for shared_memory
    assert(meta->shared_memory == 0);
    assert(meta->number_of_threads == 0);
    return { nullptr, 0, 0, 0 };
  }
  return { kernel_start_ + meta->offset,
           meta->image_size,
           int(meta->shared_memory),
           dim3 { meta->number_of_threads, 1, 1 } };
}

}
