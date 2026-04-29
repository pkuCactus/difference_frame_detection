#include <rknn_api.h>

extern "C" {

int rknn_init(rknn_context* context, void* p_mdel, uint32_t size, uint32_t flag, rknn_init_extend* extend) {
    return -1;
}

int rknn_destroy(rknn_context context) {
    return 0;
}

int rknn_query(rknn_context context, rknn_query_cmd cmd, void* info, uint32_t size) {
    return -1;
}

int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[]) {
    return -1;
}

int rknn_run(rknn_context context, rknn_run_extend* extend) {
    return -1;
}

int rknn_outputs_get(rknn_context context, uint32_t n_outputs, rknn_output outputs[], rknn_output_extend* extend) {
    return -1;
}

int rknn_outputs_release(rknn_context context, uint32_t n_outputs, rknn_output outputs[]) {
    return 0;
}

}
