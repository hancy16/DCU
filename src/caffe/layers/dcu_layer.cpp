#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/dcu_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
namespace caffe {

template <typename Dtype>
void DCULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_size = this->layer_param_.dcu_param().kernel_size();
  CHECK(kernel_size)
      << "kernel size should be larger than 0.";
  vector<int> weight_shape(1);
  weight_shape[0] = kernel_size*kernel_size;
  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  //FillerParameter filler_param(param.filler());
  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(this->layer_param_.dcu_param().weight_filler()));
  filler->Fill(this->blobs_[0].get());
}

template <typename Dtype>
void DCULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   CHECK_EQ(bottom.size(), 2) 
        << "Input bottom should be 2.";
   vector<int> top_shape = bottom[0]->shape();
   CHECK_EQ(top_shape[1] % (kernel_size*kernel_size), 0) 
       << "Bottom channel should be interger times of kernel size.";
   vector<int> bottom1_shape = bottom[1]->shape(); 
   CHECK_EQ(top_shape[2] + top_shape[3] + 2*(kernel_size -1), bottom1_shape[2] + bottom1_shape[3]) 
       << "Bottom1.shape should be equal to bottom0.shape + kernel_size -1";
   CHECK_EQ(bottom1_shape[1],top_shape[1]) 
       << "Bottom1.channel should be equal to bottom0.channel.";
// enlarge the H and W by kernel_size - 1
   top_shape[2] += kernel_size -1;
   top_shape[3] += kernel_size - 1;
   shift_data.Reshape(top_shape);
//output channel should be input channel/(kernel_size*kernel_size)
   top_shape[1] /= (kernel_size*kernel_size);
   top[0]->Reshape(top_shape);
   if (kernel_size == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
//   LOG(INFO) << "top_shape" << top_shape[1];
}

template <typename Dtype>
void DCULayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if(kernel_size == 1) {return;}
  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> bottom1_shape = bottom[1]->shape();
  vector<int> shift_shape = shift_data.shape();
  vector<int> top_shape = top[0]->shape();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom1_data = bottom[1]->cpu_data();
 // const Dtype* weight = this->blobs_[0]->cpu_data();
//  Dtype* tmp_data = shift_data.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int offset_b,offset_t,offset_h,offset_w,offset_b1;
  int kernel_total = kernel_size*kernel_size;
  Dtype tem;
//  CPUTimer timer;
//  timer.Start();
 // Dtype weight_exp[kernel_total];
 // caffe_exp<Dtype>(kernel_total,weight,weight_exp);
  caffe_set(top[0]->count(),(Dtype(0)),top_data);
 // weight_sum =0;
 // for(int i =0;i<this->blobs_[0]->count();i++)
 //      weight_sum += weight_exp[i];
// shift data in shift_data
  for(int n = 0;n < shift_shape[0];n++) {
     for(int c = 0 ;c < shift_shape[1];c++) {
        for(int h = 0 ;h < shift_shape[2];h++) {
          for(int w = 0 ;w < shift_shape[3];w++) {
            offset_h = (c%kernel_total)/kernel_size;
            offset_w = (c%kernel_total)%kernel_size;

            if(h-offset_h>=0 && h-offset_h < bottom_shape[2] && w-offset_w>=0 && w-offset_w < bottom_shape[3]) {
               offset_b = ((n * bottom_shape[1] + c) * bottom_shape[2] + h-offset_h) * bottom_shape[3] + w-offset_w;
               offset_b1 = ((n * bottom1_shape[1] + c) * bottom1_shape[2] + h) * bottom1_shape[3] + w;
               tem = bottom_data[offset_b]*bottom1_data[offset_b1];
               }
            else
               tem = 0;
            //add to top data
            offset_t = ((n * top_shape[1] + c/kernel_total) * top_shape[2] + h) * top_shape[3] + w;
            top_data[offset_t] += tem;
            
            
           } 
        }
     }
  }
//  timer.Stop();
//   LOG(INFO) << "Time for forward: " <<timer.MilliSeconds()<<"ms";
}

template <typename Dtype>
void DCULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
     vector<int> bottom_shape = bottom[0]->shape();
     vector<int> shift_shape = shift_data.shape();
     vector<int> top_shape = top[0]->shape();
     vector<int> bottom1_shape = bottom[1]->shape();
     const Dtype* top_diff = top[0]->cpu_diff();
     int offset_b,offset_t,offset_h,offset_w,offset_b1;
     int kernel_total = kernel_size*kernel_size;
   //  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
     const Dtype* bottom_data = bottom[0]->cpu_data();
     const Dtype* bottom1_data = bottom[1]->cpu_data();
    // const Dtype* weight = this->blobs_[0]->cpu_data();
    // Dtype weight_exp[kernel_total];
     //int weight_sum = 0;
     //const Dtype* top_data = bottom[0]->cpu_data();
    // caffe_exp<Dtype>(kernel_total,weight,weight_exp);
   //  for(int i =0;i<this->blobs_[0]->count();i++)
   //    weight_sum += weight_exp[i];
    // for (int i = 0; i < bottom.size(); ++i) {

      //  if (propagate_down[i]) {
            Dtype* bottom1_diff = bottom[1]->mutable_cpu_diff();
            caffe_set(bottom[1]->count(),(Dtype(0)),bottom1_diff);
          for(int n = 0;n < shift_shape[0];n++) {
	     for(int c = 0 ;c < shift_shape[1];c++) {
		for(int h = 0 ;h < shift_shape[2];h++) {
		  for(int w = 0 ;w < shift_shape[3];w++) {
		    offset_h = (c%kernel_total)/kernel_size;
		    offset_w = (c%kernel_total)%kernel_size;
                    offset_t = ((n * top_shape[1] + c/kernel_total) * top_shape[2] + h) * top_shape[3] + w;
		    if(h-offset_h>=0 && h-offset_h < bottom_shape[2] && w-offset_w>=0 && w-offset_w < bottom_shape[3]) {
		       offset_b = ((n * bottom_shape[1] + c) * bottom_shape[2] + h-offset_h) * bottom_shape[3] + w-offset_w;
                       offset_b1 = ((n * bottom1_shape[1] + c) * bottom1_shape[2] + h) * bottom1_shape[3] + w;
		      // bottom_diff[offset_b] = top_diff[offset_t]/kernel_total;
                       bottom1_diff[offset_b1] += top_diff[offset_t] * bottom_data[offset_b];
		       }
                  //  else{
                   //   offset_b = ((n * bottom_shape[1] + c) * bottom_shape[2] + h-offset_h) * bottom_shape[3] + w-offset_w;
		      // bottom_diff[offset_b] = top_diff[offset_t]/kernel_total;
                     //  bottom1_diff[offset_b1] += top_diff[offset_t] * 0;
                     //  }
		    //add to top data
		   		    
		   } 
		}
	     }
	  }
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            caffe_set(bottom[0]->count(),(Dtype(0)),bottom_diff);
            for(int n = 0;n < shift_shape[0];n++) {
	     for(int c = 0 ;c < shift_shape[1];c++) {
		for(int h = 0 ;h < shift_shape[2];h++) {
		  for(int w = 0 ;w < shift_shape[3];w++) {
		    offset_h = (c%kernel_total)/kernel_size;
		    offset_w = (c%kernel_total)%kernel_size;
                    offset_t = ((n * top_shape[1] + c/kernel_total) * top_shape[2] + h) * top_shape[3] + w;
		    if(h-offset_h>=0 && h-offset_h < bottom_shape[2] && w-offset_w>=0 && w-offset_w < bottom_shape[3]) {
		       offset_b = ((n * bottom_shape[1] + c) * bottom_shape[2] + h-offset_h) * bottom_shape[3] + w-offset_w;
                       offset_b1 = ((n * bottom1_shape[1] + c) * bottom1_shape[2] + h) * bottom1_shape[3] + w;
		       bottom_diff[offset_b] = top_diff[offset_t]*bottom1_data[offset_b1];
		       }
		    //add to top data
		   		    
		   } 
		}
	     }
	  }
     //  }
    
 //    }
}

#ifdef CPU_ONLY
STUB_GPU(DCULayer);
#endif

INSTANTIATE_CLASS(DCULayer);
REGISTER_LAYER_CLASS(DCU);

}  // namespace caffe
