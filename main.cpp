#include <opencv2/opencv.hpp>
#include <chrono>
#define px(i, j) ((Vec3b)image.at<Vec3b>(i, j));
using namespace cv;
using namespace std;
#define rows 200
#define cols 200
#define frame_l 90.0f
float x[3][rows + 2][cols + 2];
float x0[3][rows + 2][cols + 2];
float u[rows + 2][cols + 2];
float v[rows + 2][cols + 2];
float u0[rows + 2][cols + 2];
float v0[rows + 2][cols + 2];

std::chrono::steady_clock::time_point t_start;

double cur_time()
{

    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return elapsed_time_ms;
}

void set_bnd(int b, float x[][cols + 2])
{
    for (int j = 1; j <= cols; j++)
    {
        x[0][j] = b == 1 ? -x[1][j] : x[1][j];
        x[cols + 1][j] = b == 1 ? -x[cols][j] : x[cols][j];
    }
    for (int i = 1; i <= rows; i++)
    {
        x[i][0] = b == 2 ? -x[i][1] : x[i][1];
        x[i][cols + 1] = b == 2 ? -x[i][cols] : x[i][cols];
    }

    x[0][0] = 0.5f * (x[1][0] + x[0][1]);
    x[0][cols + 1] = 0.5f * (x[1][cols + 1] + x[0][cols]);
    x[rows + 1][0] = 0.5f * (x[rows][0] + x[rows + 1][1]);
    x[rows + 1][cols + 1] = 0.5f * (x[rows][cols + 1] + x[rows + 1][cols]);
}

void copy(float a[][cols + 2], float b[][cols + 2])
{
    for (int i = 1; i <= rows; i++)
    {
        for (int j = 1; j <= cols; j++)
        {
            b[i][j] = a[i][j];
        }
    }
}
void swap_arr(float a[][cols + 2], float b[][cols + 2])
{
    for (int i = 1; i <= rows; i++)
    {
        for (int j = 1; j <= cols; j++)
        {
            swap(b[i][j], a[i][j]);
        }
    }
}
void advect(float d[][cols + 2], float d0[][cols + 2], float u[][cols + 2], float v[][cols + 2], int b)
{
    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1;
    float dt0 = 1.0f;
    for (int i = 1; i <= rows; i++)
    {
        for (int j = 1; j <= cols; j++)
        {
            x = i - dt0 * u[i][j];
            y = j - dt0 * v[i][j];
            if (x < 0.5)
                x = 0.5;
            if (x > rows + 0.5)
                x = rows + 0.5;
            i0 = (int)x;
            i1 = i0 + 1;
            if (y < 0.5)
                y = 0.5;
            if (y > cols + 0.5)
                y = cols + 0.5;
            j0 = (int)y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;

            d[i][j] = s0 * (t0 * d0[i0][j0] + t1 * d0[i0][j1]) +
                      s1 * (t0 * d0[i1][j0] + t1 * d0[i1][j1]);
        }
    }
    set_bnd(b, d);
}

void project(float u[][cols + 2], float v[][cols + 2], float p[][cols + 2], float div[][cols + 2])
{
    float h;
    h = 1.0f / rows;
    for (int i = 1; i <= rows; i++)
    {
        for (int j = 1; j <= cols; j++)
        {
            div[i][j] = -0.5f * h * (u[i + 1][j] - u[i - 1][j] + v[i][j + 1] - v[i][j - 1]);
            p[i][j] = 0;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    for (int k = 0; k < 20; k++)
    {
        for (int i = 1; i <= rows; i++)
        {
            for (int j = 1; j <= cols; j++)
            {
                p[i][j] = (div[i][j] + p[i - 1][j] + p[i + 1][j] +
                           p[i][j - 1] + p[i][j + 1]) /
                          4;
            }
        }
        set_bnd(0, p);
    }
    for (int i = 1; i <= rows; i++)
    {
        for (int j = 1; j <= cols; j++)
        {
            u[i][j] -= 0.5f * (p[i + 1][j] - p[i - 1][j]) / h;
            v[i][j] -= 0.5f * (p[i][j + 1] - p[i][j - 1]) / h;
        }
    }

    set_bnd(1, u);
    set_bnd(2, v);
}

void assign_image(Mat &image)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {

            Vec3b &pixel = image.at<Vec3b>(i, j);
            for (int pi = 0; pi < 3; pi++)
            {
                pixel[pi] = min(255, (int)(x[pi][i + 1][j + 1] * 256));
            }
        }
    }
}
int clickX, clickY;
bool hasClicked = false;
void CallBackFunc(int event, int x, int y, int flags, void *userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        hasClicked = true;
        clickX = x;
        clickY = y;
    }
}
void click()
{
    int X = clickY / 2;
    int Y = clickX / 2;
    float r = (rand() % 255) / 255.0f;
    float g = 0;
    float b = (rand() % 255) / 255.0f;
    int R = 12;
    for (int i = -R; i < R; i++)
    {
        for (int j = -R; j < R; j++)
        {
            if (i * i + j * j <= R * R)
                if (X + i >= 0 && X + i <= rows + 1 && Y + j >= 0 && Y + j <= cols + 1)
                {
                    float t = (float)sqrt(i * i + j * j);

                    u[X + i][Y + j] += 4 * abs((abs(t) - R)) * i;
                    v[X + i][Y + j] += 4 * abs((abs(t) - R)) * j;

                    //  x[0][X + i][Y + j] = (t / 16) * x[0][X + i][Y + j] + (1 - t / 16) * r;
                    //  x[1][X + i][Y + j] = (t / 16) * x[1][X + i][Y + j] + (1 - t / 16) * g;
                    //   x[2][X + i][Y + j] = (t / 16) * x[2][X + i][Y + j] + (1 - t / 16) * b;
                }
        }
    }
}
void init(Mat &image)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {

            Vec3b &pixel = image.at<Vec3b>(i, j);
            x[0][i + 1][j + 1] = 0;
            //(float)pow(1.0 * i / rows, 2);

            x[0][i + 1][j + 1] = (float)pow(1.0 * sqrt((i * 1.0 - rows / 2) * (i * 1.0 - rows / 2) + (j * 1.0 - cols / 2) * (j * 1.0 - cols / 2)) / (rows * 1.0 / sqrt(2)), 1);
            //(float)pow(1.0 * j / cols, 2);

            x[2][i + 1][j + 1] = 1 - (float)pow(1.0 * sqrt((i * 1.0 - rows / 2) * (i * 1.0 - rows / 2) + (j * 1.0 - cols / 2) * (j * 1.0 - cols / 2)) / (rows * 1.0 / sqrt(2)), 1);
            float cx = rows * 0.45;
            float cy = cols * 0.45;

            if (rand() % 15 == 0)
            {
                if (rand() % 10 > 0)
                {
                    u[i + 1][j + 1] = u[i][j + 1];
                    v[i + 1][j + 1] = v[i][j];
                }
                else
                {
                    float dcx = (float)(rand() % 100 - 50);
                    float dcy = (float)(rand() % 100 - 50);
                    float dsz = sqrt(dcx * dcx + dcy * dcy);
                    if (dsz != 0)
                    {
                        u[i + 1][j + 1] = 15.0f * dcy / dsz;
                        v[i + 1][j + 1] = 15.0f * -dcx / dsz;
                    }
                }
            }
            else
            {
                u[i + 1][j + 1] = u[i + 1][j];
                v[i + 1][j + 1] = v[i + 1][j];
            }
        }
    }
}

// Diffuse the array x0 into array x
void diffuse(float x[][cols + 2], float x0[][cols + 2],
             float diff, float dt, int b)
{
    float a = diff;
    for (int k = 0; k < 20; k++)
    {
        for (int i = 1; i <= rows; i++)
        {
            for (int j = 1; j <= cols; j++)
            {
                float top = x[i - 1][j];
                float bot = x[i + 1][j];
                float left = x[i][j - 1];
                float right = x[i][j + 1];
                x[i][j] = (x0[i][j] +
                           a * (top + bot + left + right)) /
                          (1 + 4 * a);
            }
        }
    }
    set_bnd(b, x);
}

int main()
{
    srand(time(0));
    // the work...
    // Read an image from file
    Mat image(rows, cols, CV_8UC3, Scalar(255, 255, 255));

    Mat image2(2 * rows, 2 * cols, CV_8UC3, Scalar(255, 255, 255));

    // Check if the image is loaded successfully
    init(image);
    if (image.empty())
    {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }

    // Get the number of rows and columns in the image

    // Iterate over each pixel and invert the colors

    assign_image(image);

    t_start = std::chrono::high_resolution_clock::now();
    int cur_frame = 0;
    resize(image, image2, Size(2 * rows, 2 * cols), INTER_LINEAR);
    imshow("Image", image2);
    setMouseCallback("Image", CallBackFunc, NULL);
    while (true)
    {
        // Display the image
        if (hasClicked)
            click();
        hasClicked = false;
        resize(image, image2, Size(2 * rows, 2 * cols), INTER_LINEAR);
        imshow("Image", image2);
        // change velocities
        swap_arr(u, u0);
        swap_arr(v, v0);
        diffuse(u, u0, 0.5f, 0, 1);
        diffuse(v, v0, 0.5f, 0, 2);
        project(u, v, u0, v0);
        swap_arr(u, u0);
        swap_arr(v, v0);
        advect(u, u0, u0, v0, 1);
        advect(v, v0, u0, v0, 2);
        project(u, v, u0, v0);

        // change dens
        for (int i = 0; i < 3; i++)
        {
            swap_arr(x[i], x0[i]);
            diffuse(x[i], x0[i], 0.1f, 0, 0);
            swap_arr(x[i], x0[i]);
            advect(x[i], x0[i], u, v, 0);
        }
        assign_image(image);
        cur_frame++;

        // Wait for next frame
        //  waitKey(0);
        int time = (int)(frame_l * cur_frame - cur_time());
        if (time > 0)
            waitKey(time);
        else
            cout << "LAG!!!!!!!" << endl;
    }

    return 0;
}