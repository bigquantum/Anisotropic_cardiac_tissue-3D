
/*
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Developed by: Hector Augusto Velasco-Perez 
@ CHAOS Lab 
@ Georgia Institute of Technology
August 07/10/2019

Special thanks to:
Dr. Flavio Fenton
Dr. Claire Yanyan Ji
Dr. Abouzar Kaboudian

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
*/

int initGL(int *argc, char **argv);
void display(void);
void render(void);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void reshape(int w, int h);
void initPixelBuffer(void);
void idle(void);
void cleanup(void);
void createAxes(void);
void drawPoints(void);
void Draw_Axes(void);
void Turn(int key, int x, int y);
