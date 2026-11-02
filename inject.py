import re
f = open('layers/modules.h', 'r')
c = f.read()
f.close()
debug_code = """        scores = apply_causal_mask(scores);
        static int calls = 0;
        if (calls++ % (50 * 12) == 0) {
            std::vector<float> h_s(scores->total_size);
            cudaMemcpy(h_s.data(), scores->data.get(), scores->total_size * sizeof(float), cudaMemcpyDeviceToHost);
            float max_s = -1e9f;
            for (float s : h_s) if (s > max_s && s > -1e8f) max_s = s;
            std::cout << "[Softmax Telemetry] Max Pre-Softmax Logit: " << max_s << std::endl;
        }"""
c = c.replace('scores = apply_causal_mask(scores);', debug_code)
f = open('layers/modules.h', 'w')
f.write(c)
f.close()
