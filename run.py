# Importing the kivy application dependencies
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.config import Config
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '500')

# Importin the GEAR Predictor file
from runGEARP import *

#Creating the object for GEAR predictor
obj = GEARP()

# Kivy Application
class GEAR(App):

    def update_label(self):
        self.lb1.text = self.formula

    def doProcessing(self, instance):
        # Code for doing the processing
        if(instance.text == 'REALTIME PROCESSING'):
            obj.realTimePlayer()
        elif(instance.text == 'BATCH PROCESSING LOW QUALITY IMAGES'):
            obj.runGEARlowQ()
        elif(instance.text == 'BATCH PROCESSING HIGH QUALITY IMAGES'):
            obj.runGEAR()

        # Code for displaying the process done
        self.formula = 'DONE '+str(instance.text)
        self.update_label()        


    def build(self):

        # Initial box architecture
        self.formula = "GEAR PREDICTOR"
        b1 = BoxLayout(orientation='vertical', padding=10)
        g1 = GridLayout(rows = 5, spacing=0, size_hint=(1, .6))

        self.lb1 = Label(text='GEAR PREDICTOR ', font_size=20,  halign='center', valign='center', size_hint=(1, .2), text_size=(400-50, 500* .4-50))
        b1.add_widget(self.lb1)

        # Adding buttons
        g1.add_widget(Button(text="REALTIME PROCESSING", on_press = self.doProcessing,
                             background_color=[1, 0, 0, 1])), 
        g1.add_widget(Button(text="BATCH PROCESSING LOW QUALITY IMAGES", on_press = self.doProcessing, 
                             background_color=[0, 1, 0, 1]))
        g1.add_widget(Button(text="BATCH PROCESSING HIGH QUALITY IMAGES", on_press = self.doProcessing,
                             background_color=[0, 0, 1, 1]))
        b1.add_widget(g1)
        return b1
        
if __name__ == "__main__":
    GEAR().run()