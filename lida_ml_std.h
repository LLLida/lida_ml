/* Basic operations for building neural networks */
#ifndef LIDA_ML_BASIC_H
#define LIDA_ML_BASIC_H

#include "lida_ml.h"

#ifdef __cplusplus
extern "C" {
#endif

const struct lida_Gate* lida_gate_plus();
/* element-wise multiplication */
const struct lida_Gate* lida_gate_mul();
/* multiplies vector by matrix */
const struct lida_Gate* lida_gate_mm();

const struct lida_Gate* lida_gate_relu();
const struct lida_Gate* lida_gate_sigmoid();
const struct lida_Gate* lida_gate_tanh();

void lida_MSE_loss(struct lida_Loss* loss);
void lida_Cross_Entropy_Loss(struct lida_Loss* loss);

#ifdef __cplusplus
}
#endif

#endif // LIDA_ML_BASIC_H
