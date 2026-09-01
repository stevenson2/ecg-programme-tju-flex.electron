#include <math.h>
#include <string.h>
#include "respiration_ecg/respiration_ecg.h"

/**
 * @file respiration_ecg.cpp
 * @brief 呼吸阻抗辅助 ECG 去呼吸漂移实现
 *
 * 采用归一化 LMS，参考信号为呼吸阻抗通道的交流分量：
 *   ref = lowpass(resp) - baseline
 *   estimate = w * ref
 *   corrected = ecg - estimate
 *   w += mu * ref * corrected / (ref_power + eps)
 *
 * 由于呼吸信号是慢变基线干扰，而 QRS 是快速成分，该自适应过程主要
 * 收敛到呼吸相关性分量，不会明显削弱 QRS。
 */

#define REF_LP_ALPHA      0.05f    /* 呼吸参考低通 (~4Hz) */
#define REF_BASE_ALPHA    0.0005f  /* 基线估计，时间常数约 4s */
#define LMS_MU            0.01f    /* 归一化 LMS 步长 */
#define COEFF_LIMIT       10.0f    /* 防止系数发散 */
#define POWER_EPS         1e-12f

static float s_refLp     = 0.0f;
static float s_refBase   = 0.0f;
static float s_refPower  = 0.0f;
static float s_coeff     = 0.0f;
static bool  s_hasRef    = false;

void respEcgCancelInit(void)
{
    respEcgCancelReset();
}

void respEcgCancelReset(void)
{
    s_refLp = 0.0f;
    s_refBase = 0.0f;
    s_refPower = 0.0f;
    s_coeff = 0.0f;
    s_hasRef = false;
}

RespEcgCancelResult respEcgCancelProcess(float ecgRaw, float respRef)
{
    /* 呼吸参考低通 + 慢基线去除 */
    s_refLp += REF_LP_ALPHA * (respRef - s_refLp);
    s_refBase += REF_BASE_ALPHA * (s_refLp - s_refBase);
    float ref = s_refLp - s_refBase;

    /* 参考功率估计 */
    s_refPower = 0.999f * s_refPower + 0.001f * (ref * ref);
    float norm = s_refPower + POWER_EPS;

    /* 当前估计的呼吸干扰 */
    float estimate = s_coeff * ref;
    float corrected = ecgRaw - estimate;

    /* 归一化 LMS 更新 */
    if (s_hasRef && s_refPower > 1e-10f) {
        float update = LMS_MU * ref * corrected / norm;
        s_coeff += update;
        if (s_coeff > COEFF_LIMIT) s_coeff = COEFF_LIMIT;
        if (s_coeff < -COEFF_LIMIT) s_coeff = -COEFF_LIMIT;
    }
    s_hasRef = true;

    RespEcgCancelResult r;
    r.correctedEcg = corrected;
    r.estimate = estimate;
    r.coeff = s_coeff;
    r.active = (s_refPower > 1e-10f);
    return r;
}
