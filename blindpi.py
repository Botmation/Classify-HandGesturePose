import RPi.GPIO as GPIO
import time
import paho.mqtt.client as mqtt
 
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
coil_A_1_pin = 4 # pink
coil_A_2_pin = 17 # orange
coil_B_1_pin = 23 # blue
coil_B_2_pin = 24 # yellow
''' 
# adjust if different
StepCount = 8
Seq = range(0, StepCount)
Seq[0] = [0,1,0,0]
Seq[1] = [0,1,0,1]
Seq[2] = [0,0,0,1]
Seq[3] = [1,0,0,1]
Seq[4] = [1,0,0,0]
Seq[5] = [1,0,1,0]
Seq[6] = [0,0,1,0]
Seq[7] = [0,1,1,0]'''
# adjust if different
StepCount = 4
Seq = range(0, StepCount)
Seq[0] = [1,0,0,0]
Seq[1] = [0,1,0,0]
Seq[2] = [0,0,1,0]
Seq[3] = [0,0,0,1]

 
#GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(coil_A_1_pin, GPIO.OUT)
GPIO.setup(coil_A_2_pin, GPIO.OUT)
GPIO.setup(coil_B_1_pin, GPIO.OUT)
GPIO.setup(coil_B_2_pin, GPIO.OUT)
 
#GPIO.output(enable_pin, 1)
 
#Mqtt setup
MQTT_SERVER = "localhost"
MQTT_TOPIC = "blinds/position"
global server_msg = "noresults"
 

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
 
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_TOPIC)
 
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
	global server_msg
	print(msg.topic+" "+str(msg.payload))
	server_msg = str(msg.payload)
	
    
def setStep(w1, w2, w3, w4):
    GPIO.output(coil_A_1_pin, w1)
    GPIO.output(coil_A_2_pin, w2)
    GPIO.output(coil_B_1_pin, w3)
    GPIO.output(coil_B_2_pin, w4)
 
def forward(delay, steps):
    for i in range(steps):
        for j in range(StepCount):
            setStep(Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
            time.sleep(delay)
 
def backwards(delay, steps):
    for i in range(steps):
        for j in reversed(range(StepCount)):
            setStep(Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
            time.sleep(delay)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
 
client.connect(MQTT_SERVER, 1883, 60)
client.loop_start() #Listen for messages from server

if __name__ == '__main__':
    while True:
        global server_msg
        setStep(0,0,0,0) 
        steps = 50000
        delay = 5
        if server_msg == "open":
            forward(int(delay) / 1000.0, int(steps))
            server_msg = ""
		if server_msg = "close":
            backwards(int(delay) / 1000.0, int(steps))
            server_msg = ""
    
    '''
		delay = raw_input("Time Delay (ms)?")
		#you can set the stepper to idle, because otherwise power is consumed all the time (by the electromagnets in the motor)
		setStep(0,0,0,0) 
		steps = raw_input("How many steps forward? ")
		forward(int(delay) / 1000.0, int(steps))
		setStep(0,0,0,0)
		steps = raw_input("How many steps backwards? ")
		backwards(int(delay) / 1000.0, int(steps))
    '''