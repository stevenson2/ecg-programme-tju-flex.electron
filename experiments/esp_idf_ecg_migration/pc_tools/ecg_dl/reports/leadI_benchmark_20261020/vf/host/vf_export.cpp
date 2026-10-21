/* Appended to vf_detect.cpp so this translation unit can call the static
 * feature/score functions without modifying the firmware sources. */
extern "C" void vf_host_score(const float* src, float* feat, float* score, float* rms, int* suspect) {
    VfJobResult out;
    compute_features(src, feat);
    score_window(src, &out);
    *score = out.score;
    *rms = out.rms;
    *suspect = out.suspect ? 1 : 0;
}
