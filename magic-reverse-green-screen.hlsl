uniform int screen_width = 480;
uniform int screen_height = 640;

#define y_bucket_size 8
#define u_bucket_size 2
#define v_bucket_size 2

// Convert linear sRGB to BT.709 YUV
// Source: https://en.wikipedia.org/wiki/YUV#Conversion_to/from_RGB
float3 rgb_to_yuv(float3 rgb) {
    return mul(float3x3(
        0.2126, 0.7152, 0.0722,
        -0.09991, -0.33609, 0.436,
        0.615, -0.55861, -0.05639), rgb);
}

float gauss_4x4(int x, int y) {
    // Source: https://en.wikipedia.org/wiki/Gaussian_blur#Sample_Gaussian_matrix
    const float table[4][4] = {
        {0.22508352, 0.11098164, 0.01330373, 0.00038771},
        {0.11098164, 0.05472157, 0.00655965, 0.00019117},
        {0.01330373, 0.00655965, 0.00078633, 0.00002292},
        {0.00038771, 0.00019117, 0.00002292, 0.00000067},
    };
    return table[abs(y)][abs(x)];
}

int3 buckets_for_pixel(float3 pixel_yuv) {
    return int3(
        int(pixel_yuv.x * (y_bucket_size - 1)),
        int(pixel_yuv.y * (u_bucket_size - 1)),
        int(pixel_yuv.z * (v_bucket_size - 1)));
}

float4 mainImage(VertData v_in) : TARGET
{
    const int kernel_width_extent = 3;
    const int kernel_width = kernel_width_extent * 2 + 1;
    const int kernel_height_extent = 3;
    const int kernel_height = kernel_height_extent * 2 + 1;

    float4 pixel;
    float3 pixel_yuv;
    float acc;
    int x;
    int y;
    int best_count;
    int z;
    int u;
    int v;
    int3 buckets;
    int3 best_bucket;

    int bucket_hits[y_bucket_size][u_bucket_size][v_bucket_size];
    for (y = 0; y < y_bucket_size; ++y) {
        for (u = 0; u < u_bucket_size; ++u) {
            for (v = 0; v < v_bucket_size; ++v) {
                bucket_hits[y][u][v] = 0;
            }
        }
    }

    float4 pixels[kernel_width][kernel_height];
    for (x = -kernel_width_extent; x <= +kernel_width_extent; ++x) {
        for (y = -kernel_height_extent; y <= +kernel_height_extent; ++y) {
            pixel = image.Sample(
                textureSampler,
                v_in.uv + float2(
                    x * 1.0 / screen_width,
                    y * 1.0 / screen_height));
            pixels[y + kernel_height_extent][x + kernel_width_extent] = pixel;
            pixel_yuv = rgb_to_yuv(pixel.rgb);
            buckets = buckets_for_pixel(pixel_yuv);
            bucket_hits[buckets.x][buckets.y][buckets.z] += 1;
        }
    }

    best_bucket = int3(0, 0, 0);
    best_count = 0;
    for (y = 0; y < y_bucket_size; ++y) {
        for (u = 0; u < u_bucket_size; ++u) {
            for (v = 0; v < v_bucket_size; ++v) {
                if (bucket_hits[y][u][v] > best_count) {
                    best_bucket = int3(y ,u, v);
                    best_count = bucket_hits[y][u][v];
                }
            }
        }
    }

    pixel = image.Sample(textureSampler, v_in.uv);
    if (all(buckets_for_pixel(rgb_to_yuv(pixel)) == best_bucket)) {
        acc = 0.0;
        for (x = -kernel_width_extent; x <= +kernel_width_extent; ++x) {
            for (y = -kernel_height_extent; y <= +kernel_height_extent; ++y) {
                pixel = pixels[y + kernel_height_extent][x + kernel_width_extent];
                if (!all(buckets_for_pixel(rgb_to_yuv(pixel)) == best_bucket)) {
                    acc += gauss_4x4(x, y);
                }
            }
        }
        return float4(image.Sample(textureSampler, v_in.uv).rgb, pow(acc / float(kernel_width * kernel_height), 0.25));
    } else {
        return pixel;
    }
}
