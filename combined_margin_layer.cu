#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/combined_margin_layer.hpp"

namespace caffe {

  template <typename Dtype> 
  __global__ void CombinedMarginLayerForward(const int n, const int dim, const Dtype* label_data,
                        Dtype* top_data, Dtype m1, Dtype m2, Dtype m3) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label_data[index]);
      Dtype cos_theta = top_data[index * dim + gt];
      cos_theta = min(cos_theta, Dtype(1.0) - Dtype(1e-4));
      cos_theta = max(cos_theta, Dtype(-1.0) + Dtype(1e-4));
      float theta = acos(cos_theta);
      Dtype m1_mul_theta_plus_m2 = m1 * theta + m2;
      if (m1_mul_theta_plus_m2 > M_PI - Dtype(1e-4)) m1_mul_theta_plus_m2 = M_PI - Dtype(1e-4);
      if (m1_mul_theta_plus_m2 < 0) m1_mul_theta_plus_m2 = 0;
      top_data[index * dim + gt] = cos(m1_mul_theta_plus_m2) -m3;
    }
  }

  template <typename Dtype> 
  __global__ void CombinedMarginLayerBackward(const int n, const int dim, const Dtype* label_data,
                        const Dtype* bottom_data, Dtype* bottom_diff, Dtype m1, Dtype m2) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label_data[index]);
      if(gt < 0) continue;
      Dtype cos_theta = bottom_data[index * dim + gt];
      cos_theta = min(cos_theta, Dtype(1.0) - Dtype(1e-4));
      cos_theta = max(cos_theta, Dtype(-1.0) + Dtype(1e-4));
      float theta = acos(cos_theta);
      Dtype m1_mul_theta_plus_m2 = m1 * theta + m2;
      if (m1_mul_theta_plus_m2 > M_PI - Dtype(1e-4)) m1_mul_theta_plus_m2 = M_PI - Dtype(1e-4);
      if (m1_mul_theta_plus_m2 < 0) m1_mul_theta_plus_m2 = 0;
      Dtype diff_gt = m1 * powf(1 - powf(cos_theta, 2), -0.5) * sin(m1_mul_theta_plus_m2);
      diff_gt = diff_gt > 2 ? 2 : diff_gt;
      diff_gt = diff_gt < 0 ? 0 : diff_gt;
      bottom_diff[index * dim + gt] *= diff_gt;
    }
  }

  template <typename Dtype>
  void CombinedMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, bottom_data, top_data);
    
    // NOLINT_NEXT_LINE(whitespace/operators)
    CombinedMarginLayerForward<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>> (
      num, dim, label_data, top_data, m1, m2, m3);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void CombinedMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    
    if (propagate_down[0])
    {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* label_data = bottom[1]->gpu_data();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      int count = bottom[0]->count();

      caffe_copy(count, top_diff, bottom_diff);

      int num = bottom[0]->num();
      int dim = count / num;

      // NOLINT_NEXT_LINE(whitespace/operators)
      CombinedMarginLayerBackward<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>> (
        num, dim, label_data, bottom_data, bottom_diff, m1, m2);
      CUDA_POST_KERNEL_CHECK;
    }
  }
  INSTANTIATE_LAYER_GPU_FUNCS(CombinedMarginLayer);
} // namespace caffe