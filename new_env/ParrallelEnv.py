import simpy
import Machine
import Spec

class   C_ParallelEnv () :
    def __init__ (self) :
        self.m_env = simpy.Environment()

        self.m_nNumOfCuringMch = 9
        self.m_nNumOfBuildingMch = 2
        self.m_nNumOfSpec = 6

        self.m_spec_lst = []
        self.m_curingMch_lst = []
        self.m_buildingMch_lst = []

        self.reset_spec ()
        self.reset_Mch ()




    def reset_spec (self) :



        spec = Spec.C_Spec (self.m_env, 'A', 1, 143, 390, 141)
        self.m_spec_lst.append (spec)
        spec = Spec.C_Spec (self.m_env, 'B', 2, 143, 445, 64)
        self.m_spec_lst.append (spec)
        spec = Spec.C_Spec (self.m_env, 'C', 3, 143, 348, 67)
        self.m_spec_lst.append (spec)
        spec = Spec.C_Spec (self.m_env, 'D', 4, 143, 480, 0)
        self.m_spec_lst.append (spec)
        spec = Spec.C_Spec (self.m_env, 'E', 5, 143, 440, 15)
        self.m_spec_lst.append (spec)
        spec = Spec.C_Spec (self.m_env, 'F', 6, 143, 480, 42)
        self.m_spec_lst.append (spec)

    def inv_monitor_generator (self) :
        while (self.m_env.now < 86400 - 1):
            yield self.m_env.timeout(1000)
            for i in range(self.m_nNumOfSpec):
                nInventory = len(self.m_spec_lst[i].m_store.items)
                print(f'time : {self.m_env.now}, name : {self.m_spec_lst[i].m_name}, inventory : {nInventory}')

    def reset_Mch (self) :
        for i in range (self.m_nNumOfCuringMch) :
            curingMch = Machine.C_CuringMch (self.m_env, f'Curing{i}')
            self.m_curingMch_lst.append (curingMch)

        for i in range(self.m_nNumOfBuildingMch):
            buildingMch = Machine.C_BuildingMch (self.m_env, f'Building{i}')
            self.m_buildingMch_lst.append (buildingMch)




    def run_simulation (self) :
        self.m_curingMch_lst[0].add_queue (self.m_spec_lst[0], 222)
        self.m_curingMch_lst[1].add_queue (self.m_spec_lst[1], 194)
        self.m_curingMch_lst[2].add_queue (self.m_spec_lst[1], 188)
        self.m_curingMch_lst[2].add_queue (self.m_spec_lst[2], 42)
        self.m_curingMch_lst[3].add_queue (self.m_spec_lst[1], 190)
        self.m_curingMch_lst[4].add_queue (self.m_spec_lst[2], 206)
        self.m_curingMch_lst[4].add_queue (self.m_spec_lst[1], 40)
        self.m_curingMch_lst[5].add_queue (self.m_spec_lst[5], 112)
        self.m_curingMch_lst[5].add_queue (self.m_spec_lst[3], 66)
        self.m_curingMch_lst[6].add_queue (self.m_spec_lst[4], 196)


        self.m_buildingMch_lst[0].add_queue(self.m_spec_lst[1], 70)
        self.m_buildingMch_lst[0].add_queue(self.m_spec_lst[0], 94)
        self.m_buildingMch_lst[0].add_queue(self.m_spec_lst[1], 411)

        self.m_buildingMch_lst[1].add_queue(self.m_spec_lst[2], 51)
        self.m_buildingMch_lst[1].add_queue(self.m_spec_lst[4], 110)
        self.m_buildingMch_lst[1].add_queue(self.m_spec_lst[5], 61)
        self.m_buildingMch_lst[1].add_queue(self.m_spec_lst[2], 100)
        self.m_buildingMch_lst[1].add_queue(self.m_spec_lst[4], 90)
        self.m_buildingMch_lst[1].add_queue(self.m_spec_lst[0], 100)
        self.m_buildingMch_lst[1].add_queue(self.m_spec_lst[3], 80)

        self.m_env.process(self.inv_monitor_generator())

        self.m_env.run (until = 86400)

        for i in range (self.m_nNumOfCuringMch) :
            for j in range (len (self.m_curingMch_lst[i].m_backOrder_lst)) :
                item, t1, t2 = self.m_curingMch_lst[i].m_backOrder_lst[j]
                print (f'machine:{i},seq:{item},delay:{t1},time:{t2}')

        for i in range (self.m_nNumOfSpec) :
            nInventory = len (self.m_spec_lst[i].m_store.items)
            print (f'name : {self.m_spec_lst[i].m_name}, inventory : {nInventory}')

prEnv = C_ParallelEnv ()
prEnv.run_simulation()
