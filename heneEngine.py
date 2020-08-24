import math
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders
import numpy
import pyrr
import random
import multiprocessing
import tkinter as tk
import socket
import threading
from PIL import Image



class Model3Dvbo():
    def __init__(self, filename, index, img_index, text_list, text_koko, nosto):

        self.filename = filename
        if self.filename == "obj/freeformU.obj":
            print("ladataan terrainia... hetki menee... 558 147 facee..")
        self.text_koko = text_koko
        self.img_index = img_index
        self.nosto = nosto
        self.verticies = []
        self.normals = []
        self.index = index  # indexi CallListia varten Ja TEX iideetä
        self.tex_koords = []
        self.pinta_kaikilla = []
        self.alustus()

        print(self.filename)
        print("poly ",len(self.pinta_kaikilla)/3," vert ", len(self.verticies),"\n")

        self.lattia_nosto(nosto)
        self.tex_koords_kerrottu = self.tex_koord_kerroin()
        self.lataa_text(text_list)

        self.vbo = self.vboize()
        #self.VerteXnormal()

        self.tangenttilista = self.tangenttitehdas()
        self.vbo = self.tangentti_mukaan_vbohon()
        self.vbo = numpy.array(self.vbo, dtype=numpy.float32)

        self.VBOVAOgenerointi()



    def alustus(self):
        self.file = open(self.filename, "r")

        for rivi in self.file:
            pygame.event.get()
            rivi_lista = rivi.split()
            try:
                tyyppi = rivi_lista[0]
                data = rivi_lista[1:]

                if tyyppi == "v":
                    x, y, z = data
                    vertex = (float(x), float(y), float(z))
                    self.verticies.append(vertex)

                elif tyyppi == "vt":
                    x = data[0]
                    y = data[1]

                    tex_coords = (float(x), float(y))
                    self.tex_koords.append(tex_coords)

                elif tyyppi == "vn":
                    x, y, z = data
                    normal = (float(x), float(y), float(z))
                    self.normals.append(normal)


                elif tyyppi == "f":
                    for v_vt_vn in data:
                        vtn = list((v_vt_vn.split("/")))
                        self.pinta_kaikilla.append(vtn)

            except:
                ValueError

        self.file.close()
    def vboize(self):
        VBO_alku = []
        VBOO = []
        self.vertexit_jarjestyksessa = []
        self.UV_jarjestyksessa = []
        self.normaalit_jarjestyksessa = []
        for alkio in self.pinta_kaikilla:
            pygame.event.get()
            v = (self.verticies[int(alkio[0]) - 1])
            vt = (self.tex_koords_kerrottu[int(alkio[1]) - 1])
            vn = (self.normals[int(alkio[2]) - 1])

            self.vertexit_jarjestyksessa.append(v)
            self.UV_jarjestyksessa.append(vt)
            self.normaalit_jarjestyksessa.append(vn)

            VBO_alku.append(v)
            VBO_alku.append(vt)
            VBO_alku.append(vn)
        for osa in VBO_alku:
            pygame.event.get()
            for OSA in osa:
                VBOO.append(OSA)
        del self.normals
        del self.tex_koords
        del self.tex_koords_kerrottu
        return VBOO
    def tangenttitehdas(self):
        #tangenttilista_palanen = []
        tangenttilista = []
        for x in range(len(self.vertexit_jarjestyksessa)):
            pygame.event.get()
            if x % 3 == 0:
                # edge1
                v1_v00 = self.vertexit_jarjestyksessa[x + 1][0] - self.vertexit_jarjestyksessa[x][0]
                v1_v01 = self.vertexit_jarjestyksessa[x + 1][1] - self.vertexit_jarjestyksessa[x][1]
                v1_v02 = self.vertexit_jarjestyksessa[x + 1][2] - self.vertexit_jarjestyksessa[x][2]

                # edge2
                v2_v00 = self.vertexit_jarjestyksessa[x + 2][0] - self.vertexit_jarjestyksessa[x][0]
                v2_v01 = self.vertexit_jarjestyksessa[x + 2][1] - self.vertexit_jarjestyksessa[x][1]
                v2_v02 = self.vertexit_jarjestyksessa[x + 2][2] - self.vertexit_jarjestyksessa[x][2]

                # deltaUV1
                uv1_uv00 = self.UV_jarjestyksessa[x + 1][0] - self.UV_jarjestyksessa[x][0]
                uv1_uv01 = self.UV_jarjestyksessa[x + 1][1] - self.UV_jarjestyksessa[x][1]

                # deltaUV2
                uv2_uv00 = self.UV_jarjestyksessa[x + 2][0] - self.UV_jarjestyksessa[x][0]
                uv2_uv01 = self.UV_jarjestyksessa[x + 2][1] - self.UV_jarjestyksessa[x][1]

                try:
                    r = 1 / ((uv1_uv00 * uv2_uv01) - (uv1_uv01 * uv2_uv00))
                except:
                    r = 1
                    print("fail")

                Tx = ((v1_v00 * uv2_uv01) - (v2_v00 * uv1_uv01)) * r
                Ty = ((v1_v01 * uv2_uv01) - (v2_v01 * uv1_uv01)) * r
                Tz = ((v1_v02 * uv2_uv01) - (v2_v02 * uv1_uv01)) * r

                Tangentti = [Tx, Ty, Tz]
                tangenttilista.append(Tangentti)
                tangenttilista.append(Tangentti)
                tangenttilista.append(Tangentti)
        del self.UV_jarjestyksessa
        del self.vertexit_jarjestyksessa
        return tangenttilista
    def tangentti_mukaan_vbohon(self):
        prosessoitu_vbo = []
        y = -1
        for x in range(len(self.vbo)):
            pygame.event.get()
            if x % 8 == 0:
                prosessoitu_vbo.append(self.tangenttilista[y][0])
                prosessoitu_vbo.append(self.tangenttilista[y][1])
                prosessoitu_vbo.append(self.tangenttilista[y][2])
                y += 1
            prosessoitu_vbo.append(self.vbo[x])
        del prosessoitu_vbo[0]
        del prosessoitu_vbo[0]
        del prosessoitu_vbo[0]
        prosessoitu_vbo.append(self.tangenttilista[-1][0])
        prosessoitu_vbo.append(self.tangenttilista[-1][1])
        prosessoitu_vbo.append(self.tangenttilista[-1][2])
        del self.tangenttilista
        return prosessoitu_vbo
    def tex_koord_kerroin(self):
        kerrotut = []
        for tex in self.tex_koords:
            pygame.event.get()
            kerrotut.append(((tex[0] / self.text_koko), (tex[1] / self.text_koko)))
        return kerrotut
    def lataa_text(self, text_list):
        y = 0
        if text_list[0] == 0:
            plane_texture = glGenTextures(1, self.img_index)
            glBindTexture(GL_TEXTURE_2D, plane_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(GL_TEXTURE_2D, 0)
        elif text_list[0] == 1:
            reflect_plane_texture = glGenTextures(1, self.img_index)
            glBindTexture(GL_TEXTURE_2D, reflect_plane_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            #glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(GL_TEXTURE_2D, 0)

            image = pygame.image.load(text_list[1])
            width = image.get_width()
            height = image.get_height()
            image = pygame.image.tostring(image, "RGBA", True)

            texture_ID = self.img_index + 1
            ID = glGenTextures(1, texture_ID)
            glBindTexture(GL_TEXTURE_2D, ID)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
            #glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        elif text_list[0] == 2:
            reflect_plane_texture = glGenTextures(1, self.img_index)
            glBindTexture(GL_TEXTURE_2D, reflect_plane_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 800, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(GL_TEXTURE_2D, 0)

        else:
            for text in text_list:
                image = pygame.image.load(text)
                width = image.get_width()
                height = image.get_height()
                image = pygame.image.tostring(image, "RGBA", True)

                texture_ID = self.img_index + y
                y += 1
                ID = glGenTextures(1, texture_ID)
                glBindTexture(GL_TEXTURE_2D, ID)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
                glGenerateMipmap(GL_TEXTURE_2D)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    def VBOVAOgenerointi(self):

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, (len(self.vbo) * 4), self.vbo, GL_STATIC_DRAW)

        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(32))
        glEnableVertexAttribArray(3)


        glBindVertexArray(0)
    def instanceBuffer(self, instanssi_mat_list, instanssi_trans_list):

        self.instance_array = numpy.array(instanssi_mat_list, dtype=numpy.float32)
        self.instance_array2 = numpy.array(instanssi_trans_list, dtype=numpy.float32)


        instanceM_VBO = glGenBuffers(1, self.index)
        glBindBuffer(GL_ARRAY_BUFFER, instanceM_VBO)
        glBufferData(GL_ARRAY_BUFFER, 64 * len(self.instance_array), self.instance_array, GL_STATIC_DRAW)

        glBindVertexArray(self.index)

        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(0))
        glEnableVertexAttribArray(4)

        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(16))
        glEnableVertexAttribArray(5)

        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(32))
        glEnableVertexAttribArray(6)

        glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(48))
        glEnableVertexAttribArray(7)

        glVertexAttribDivisor(4, 1)
        glVertexAttribDivisor(5, 1)
        glVertexAttribDivisor(6, 1)
        glVertexAttribDivisor(7, 1)

        instanceT_VBO = glGenBuffers(1, self.index)
        glBindBuffer(GL_ARRAY_BUFFER, instanceT_VBO)
        glBufferData(GL_ARRAY_BUFFER, 64 * len(self.instance_array2), self.instance_array2, GL_STATIC_DRAW)

        glBindVertexArray(self.index)

        glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(0))
        glEnableVertexAttribArray(8)

        glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(16))
        glEnableVertexAttribArray(9)

        glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(32))
        glEnableVertexAttribArray(10)

        glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(48))
        glEnableVertexAttribArray(11)

        glVertexAttribDivisor(8, 1)
        glVertexAttribDivisor(9, 1)
        glVertexAttribDivisor(10, 1)
        glVertexAttribDivisor(11, 1)

        randomlista = []
        for z in range(len(self.instance_array)):
            randomlista.append(random.random())

        randomlista = numpy.array(randomlista, dtype=numpy.float32)

        random_VBO = glGenBuffers(1, self.index)
        glBindBuffer(GL_ARRAY_BUFFER, random_VBO)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(randomlista), randomlista, GL_STATIC_DRAW)

        glBindVertexArray(self.index)

        glVertexAttribPointer(12, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(12)

        glVertexAttribDivisor(12, 1)
    def piirra(self, shader, shader_type):
        if shader_type == "no_light_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "Ishader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "normalMap")
            glUniform1i(tex, 1)

            glBindVertexArray(self.index)
            glDrawArraysInstanced(GL_TRIANGLES, 0, int(len(self.vbo) / 11), len(self.instance_array))
            glBindVertexArray(0)
        elif shader_type == "perus_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "normalMap")
            glUniform1i(tex, 1)

            glBindVertexArray(self.index)
            #glDrawElements(GL_TRIANGLES,len(self.indices),GL_UNSIGNED_INT,None)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "T_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 8)
            tex = glGetUniformLocation(shader, "blendMap")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "BGTex")
            glUniform1i(tex, 1)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 2)
            tex = glGetUniformLocation(shader, "rTex")
            glUniform1i(tex, 2)

            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 4)
            tex = glGetUniformLocation(shader, "gTex")
            glUniform1i(tex, 3)

            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 6)
            tex = glGetUniformLocation(shader, "bTex")
            glUniform1i(tex, 4)


            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "BGnormalMap")
            glUniform1i(tex, 5)

            glActiveTexture(GL_TEXTURE6)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 3)
            tex = glGetUniformLocation(shader, "RnormalMap")
            glUniform1i(tex, 6)

            glActiveTexture(GL_TEXTURE7)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 5)
            tex = glGetUniformLocation(shader, "GnormalMap")
            glUniform1i(tex, 7)

            glActiveTexture(GL_TEXTURE8)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 7)
            tex = glGetUniformLocation(shader, "BnormalMap")
            glUniform1i(tex, 8)

            glBindVertexArray(self.index)
            #glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "T2_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "normalMap")
            glUniform1i(tex, 1)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "Dshader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "S_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "normalMap")
            glUniform1i(tex, 1)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D,7)
            tex = glGetUniformLocation(shader, "shadowMap")
            glUniform1i(tex, 2)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "ref_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_index)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.img_index + 1)
            tex = glGetUniformLocation(shader, "dudvMap")
            glUniform1i(tex, 1)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "F_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 44)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 46)
            tex = glGetUniformLocation(shader, "toinen")
            glUniform1i(tex, 1)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, 48)
            tex = glGetUniformLocation(shader, "prev1")
            glUniform1i(tex, 2)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "F2_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 44)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 46)
            tex = glGetUniformLocation(shader, "toinen")
            glUniform1i(tex, 1)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "hor_blur_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 47)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "ver_blur_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 45)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
        elif shader_type == "bs_shader":
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 44)
            tex = glGetUniformLocation(shader, "samplerTexture")
            glUniform1i(tex, 0)

            glBindVertexArray(self.index)
            glDrawArrays(GL_TRIANGLES, 0, int(len(self.vbo) / 11))
            glBindVertexArray(0)
    def lattia_nosto(self, nosto):
        noni = []
        for vert in self.verticies:
            pygame.event.get()
            asd = vert[1] + nosto
            uudet = [vert[0], asd, vert[2]]
            noni.append(uudet)
            uudet = []
        self.verticies = noni
    def VerteXnormal(self):
        self.vertexnormal = []
        for gynther in range(len(self.vertexit_jarjestyksessa)):
            self.vertexnormal.append(self.vertexit_jarjestyksessa[gynther])
            self.vertexnormal.append(self.normaalit_jarjestyksessa[gynther])
        del self.normaalit_jarjestyksessa

class ASETUSARVOJA():
    def __init__(self):
        self.P = 0
        self.V = 0
        self.Vlast = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0]))
        self.Lpos = 0
        self.Vpos = 0
        self.Vref = 0

        self.lattia_skaala = 0.3


        self.zoom = 0
        self.ZOOM = 0

        self.display = (1280, 800)
        self.test_rot = 0
        self.kamera_addon = []

        self.Max_nopeus = 7
        self.Kiihtyvyys = 0.01 #ei vaikuta kiihtyvyyteen mitenkaan vaan nopeuteen :D

        self.XZ_move_ws = 0
        self.XZ_MOVE_ws = 0
        self.XZ_move_vektori_ws = [0.0, 0.0]

        self.XZ_move_ad = 0
        self.XZ_MOVE_ad = 0
        self.XZ_move_vektori_ad = [0.0, 0.0]

        self.moving = False

        self.hahmo_koords = [5.0, -3.0, 5.0]
        self.lattia_grids = [[], []]  # tahan montako lattiaa :D vittumita kurapaskakoodia -- EI toimi multipaskan kaa
        self.Y_kiihtyvyys = 0.003
        self.Y_vektori = 0
        self.lattiassa_kiinni = True
        self.jump = 0.1
        self.deny_jump = False

        self.sens = 0.08
        self.y_rotation = 314096.8
        self.x_rotation = 314311.4
        self.mousepos_edellinen_x = 640
        self.mousepos_edellinen_y = 400
        self.invert_y = False

        self.luhtu = 0
        self.luhtu_strena = 1.5

        self.cube1_pos = [46.0, 4.5, 2.0]
        self.cube2_pos = [49.0, 10.0, 7.0]
        self.cube3_pos = [44.0, 3.0, 8.0]
        self.cube4_pos = [40.0, 7.0, 0.0]

        self.possibleC = [self.cube1_pos, 1,1 , self.cube2_pos, 1, 1, self.cube3_pos, 1, 1,self.cube4_pos,1,1, [48, 1, 5], 0.7, 1]
        self.collision = False

        self.cube_pos_list = [self.objposMatrix(self.cube1_pos), self.objposMatrix(self.cube2_pos),
                              self.objposMatrix(self.cube3_pos),self.objposMatrix(self.cube4_pos)]
        self.lattia_pos = self.objposMatrix([0.0, 0.5, 0.0])
        self.lattia2_pos = self.objposMatrix([0.0, 0.0, 0.0])
        self.pallo_pos = self.objposMatrix([48, 1, 5])

        self.depth_screen = False
        self.disto_move_speed = 0.002
        self.disto_move = 0

        self.reflect = True
        self.PP = 0
        self.MB = 0
        self.terrain_Mtex_Mnorm = 0
        self.GLOW = 0
        self.AL = 0.0
        self.SL = 1.0
        self.kasvi = True

        self.Client2pos = [0.0,-10.0,0.0]

        self.Evolume = 0.1
        self.Mvolume = 0.3



    def objposMatrix(self, pos):
        objpos = pyrr.matrix44.create_from_translation(pyrr.Vector3(pos))
        return objpos

class kiekko:
    def __init__(self,speed,glide,turn,fade):
        self.speed = speed
        self.glide = glide
        self.turn = turn
        self.fade = fade
        self.paikkavektori = [0.0,0.0]
        self.nopeus_vektori = []
        self.alkunopeus_vektori = []
        self.kiihtyvyys_vektori = []
        self.paino = 0.175
        self.A = 0.003
        self.ilmanvastuskerroin =0
        self.timestep = 0.08
        self.turni = 0.0
        self.turn_addon = 0
        self.fade_addon = 0
        self.turn_addon_h = 0
        self.fade_addon_h = 0
        self.speed_addon = 0
        self.korkeus = 2
        self.gravity = 1
        self.Kkulma = 0
        self.hyzerkulma = 0
        self.heitto = False

    def IVK(self):
        ilmanvastuskerroin = 1.0 + 2.0 / (self.speed+self.speed_addon)
        return ilmanvastuskerroin

    def nopeus(self):
        nopeus = math.sqrt(self.nopeus_vektori[0]**2+self.nopeus_vektori[1]**2)
        return nopeus

    def ilmanvastus(self):
        ilmanvastus= ((0.5*1.3*self.nopeus()**2)*self.A*self.IVK())/self.paino
        return ilmanvastus

    def kiihtyvyys(self):
        X_kiihtyvyys = -(self.ilmanvastus()*self.yksikkovektori()[0])
        Z_kiihtyvyys = -(self.ilmanvastus()*self.yksikkovektori()[1])
        return [X_kiihtyvyys,Z_kiihtyvyys]

    def yksikkovektori(self):
        yksikkovektori = [(self.nopeus_vektori[0]/self.nopeus()),(self.nopeus_vektori[1]/self.nopeus())]
        return yksikkovektori

    def paikka(self):

        if self.speed > 4:
            if self.turn == 0:
                self.turn = -0.4
            if self.fade == 0:
                self.fade = 1
        elif self.speed <= 4:
            if self.turn == 0:
                self.turn = -0.1
            if self.fade > 1:
                self.fade = 1
        elif self.speed > 4 and self.speed < 6:
            if self.fade > 1:
                self.fade = 1+self.fade/10


        self.turni = (self.nopeus()/10* 1/(self.speed) * -(self.turn+self.turn_addon+self.turn_addon_h)/(self.speed)* 2) - (1/(self.speed) * (self.fade+self.fade_addon+self.fade_addon_h)/self.nopeus())


        self.nopeus_vektori_pyrr = pyrr.Vector3([self.nopeus_vektori[0],self.nopeus_vektori[1],0])
        self.vektori_pyrr = pyrr.Vector3([self.nopeus_vektori[0],self.nopeus_vektori[1],1])
        self.normal = pyrr.vector.normalise(pyrr.Vector3.cross(self.nopeus_vektori_pyrr,self.vektori_pyrr))



        self.nopeus_vektori[0] -= self.normal[0]*self.turni
        self.nopeus_vektori[1] -= self.normal[1]*self.turni


        self.nopeus_vektori[0] = self.yksikkovektori()[0] * self.nopeus()
        self.nopeus_vektori[1] = self.yksikkovektori()[1] * self.nopeus()

        self.kiihtyvyys_vektori=[self.kiihtyvyys()[0],self.kiihtyvyys()[1]]


        self.nopeus_vektori[0] += (self.kiihtyvyys_vektori[0]* self.timestep)
        self.nopeus_vektori[1] += (self.kiihtyvyys_vektori[1]* self.timestep)


        steppimatka = math.sqrt(((self.nopeus_vektori[0] * self.timestep)**2)+((self.nopeus_vektori[1] * self.timestep)**2))
        steppinousu = steppimatka * self.Kkulma/10
        self.korkeus += steppinousu
        self.korkeus-=(self.gravity/5)*abs(self.Kkulma/3)
        #self.korkeus -= abs(self.turni)/10



        self.paikkavektori[0] += (self.nopeus_vektori[0] * self.timestep)
        self.paikkavektori[1] += (self.nopeus_vektori[1] * self.timestep)

        #print(self.turni)
        paikka = [self.paikkavektori[0], self.paikkavektori[1]]
        return paikka,self.korkeus

    def hyzer(self,kulma):
        if self.speed <= 4:
            kulma = (kulma*self.speed)/13
        if self.speed > 4 and self.speed < 9:
            kulma=kulma*self.speed/7
        if self.speed >= 9:
            kulma=kulma*self.speed/5
        if kulma >= 0:
            self.fade_addon += kulma/10
            self.turn_addon += kulma/10
        else:
            self.turn_addon += kulma/10

    def korkeuskulma(self,kulma):
        if self.speed <= 4:
            kulma = (kulma*self.speed)/13
        if self.speed > 4 and self.speed < 9:
            kulma=kulma*self.speed/13
        if self.speed >= 9:
            kulma=kulma*self.speed/13
        if kulma >= 0:
            self.fade_addon_h += kulma/10
            self.turn_addon_h += kulma/10
        else:
            self.turn_addon_h += kulma/10

        self.speed_addon = -kulma
        return kulma

    def korkeuskiihtyvyys(self):
        KK = self.gravity-((-0.0035*self.nopeus()**2)+(0.1665*self.nopeus())-0.8633)
        return KK


def main(jono, hk0, hk1, hk2, sulje, jono2, hahmoY, collision, Ctype, deny_jump,hcy,c2x,c2y,c2z):
    asetusarvo = ASETUSARVOJA()
    init_window(asetusarvo)
    Clock = pygame.time.Clock()  # fps ticks

    #COMPILEE SHADERIT
    perus_shader = normi_shader()
    no_light_shader=nolight_shader()
    Ishader = instanssi_shaderi()
    Irefshader = instanssi_shaderi_ref()
    loading_shader=loading_shaiba()
    Dshader = depth_shader()
    #S_shader = shadow_shader()
    ref_shader = reflect_shader()
    F_shader = final_shader()
    hor_blur_shader = hor_blur()
    ver_blur_shader = ver_blur()
    bs_shader = brightspot_shader()
    F2_shader = motionblur_shader()
    T_shader = terrain_shader()
    T2_shader = terrain_nomulti_shader()
    I2shader = instanssi_shaderi_wind()
    I2refshader = instanssi_shaderi_wind_ref()

    # objekti, objekti_indexi, image_indexi..t, [teks ja normal], tekstuuri multiplier, manuaalinen koordinaatien Y siirto
    draw_loading(loading_shader)
    kolmio = Model3Dvbo("obj/cubeUV.obj", 2, 1, ["text/cubemaptest.png", "nrml/cubemaptest_normal.png"], 1, 0)
    draw_loading(loading_shader)
    lattia = Model3Dvbo("obj/hervanta2.obj", 3, 3, ["terrain/BGFLOOR.jpg", "terrain/GRASSNORMAL.jpg", "terrain/SAND.jpg", "terrain/SANDNORMAL.jpg", \
                         "terrain/grass_texture.png", "terrain/SANDNORMAL.jpg", "terrain/GRAVELL.jpg",
                         "terrain/GRAVELNORMAL.jpg", "terrain/hervantaBlend.png"], 1, 0)
    draw_loading(loading_shader)
    pallo = Model3Dvbo("obj/pallo.obj", 4, 12, ["text/carbon.png", "nrml/rock_wall_normal.png"], 2, 0)
    draw_loading(loading_shader)
    plane = Model3Dvbo("obj/plane6.obj", 5, 14, [0], 1, 0)
    draw_loading(loading_shader)
    vesi = Model3Dvbo("obj/vesi.obj", 6, 15, ["text/laava2.png", "nrml/laava_normal2.png"], 10, 0)
    draw_loading(loading_shader)
    plant = Model3Dvbo("obj/hightext_brantsi.obj", 7, 17, ["text/brantsi_lehti2.png", "nrml/brantsi_lehti_normal.png"], 1, 0)
    draw_loading(loading_shader)
    skybox = Model3Dvbo("obj/skybox.obj", 8, 19, ["text/clouds3.jpg", "nrml/rock_wall_normal.png"], 1, 0)
    draw_loading(loading_shader)
    rock = Model3Dvbo("obj/blenderkivi.obj", 9, 21, ["text/rocktex.png", "nrml/laava_normal.png"], 10, 0)
    draw_loading(loading_shader)
    runko = Model3Dvbo("obj/runko.obj", 10, 23, ["text/rocktex.png", "nrml/laava_normal.png"], 1, 0)
    draw_loading(loading_shader)
    reflect_plane = Model3Dvbo("obj/reflect_surface.obj", 11, 25, [1,"dudv/dudv2.jpg"], 1, 0)
    draw_loading(loading_shader)
    plane2 = Model3Dvbo("obj/plane6.obj", 12, 27, [2], 1, 0)
    draw_loading(loading_shader)
    main_screen = Model3Dvbo("obj/Mscr.obj", 13, 28, [2], 1, 0)
    instacube = Model3Dvbo("obj/cubeUV.obj", 14, 29, ["text/carbon.png", "nrml/cubemaptest_normal.png"], 1, 0)

    lattia_ref = Model3Dvbo("obj/hervantaLow.obj", 15, 31,
                        ["terrain/BGFLOOR.jpg", "terrain/GRASSNORMAL.jpg", "terrain/SAND.jpg", "terrain/SANDNORMAL.jpg", \
                         "terrain/grass_texture.png", "terrain/SANDNORMAL.jpg", "terrain/GRAVELL.jpg",
                         "terrain/GRAVELNORMAL.jpg", "terrain/hervantaBlend.png"], 1, 0)

    kasvi = Model3Dvbo("obj/blenderkasvi.obj", 16, 40, ["text/hightext_kasvi.png", "nrml/hightext_kasvi_normal.png"], 1, 0)
    pine = Model3Dvbo("obj/blenderkuusi.obj", 17, 42, ["text/kuusitextpohja.png", "nrml/kuusinormal.png"], 1, 0)


    ########### lattiagridit ######### koska prosessorin saastaminen
    lattia_lista = [lattia]#, lattia2]
    lattia_grid(asetusarvo, lattia_lista)

    draw_loading(loading_shader)
    ################### instanssi objektit #####################
    instacounts = instanssi_saadot(lattia, asetusarvo, loading_shader, rock, plant, runko, instacube, kasvi,pine)

    vertexcount = ((len(kolmio.pinta_kaikilla) * 3*2) + (len(lattia.pinta_kaikilla) * 1) + (len(pallo.pinta_kaikilla) * 2)+(len(plant.pinta_kaikilla) * instacounts[1]*2)+\
                  (len(rock.pinta_kaikilla) * instacounts[0]*2)+(len(runko.pinta_kaikilla) * instacounts[2]*2)+(len(kasvi.pinta_kaikilla) * instacounts[3]*2)+\
                  (len(instacube.pinta_kaikilla) * instacounts[4]*2)+(len(lattia_ref.pinta_kaikilla)+(len(pine.pinta_kaikilla) * instacounts[5]*2)))/3
    print("POLYGONEJA n. ",vertexcount)

    #tehaan tarvittavatFBOt
    fbo_depth = FBO_depth(plane)
    fbo_reflect = FBO_reflect(reflect_plane)
    fbo_plane2 = FBO_plane2_render(plane2)
    fbo_screen = FBOscreen()
    fbo_hor_blur_screen = FBO_hor_blur_screen()
    fbo_ver_blur_screen = FBO_ver_blur_screen()
    fbo_brightspot = FBO_brightspot()
    fbo_prevframe1 = FBO_prevframe1()

    #for Xx in range(30000):
        #asetusarvo.possibleC.append([0,0,0])
        #asetusarvo.possibleC.append(1)
        #asetusarvo.possibleC.append(1)
    #print(len(asetusarvo.possibleC))


    jono.put(asetusarvo.lattia_grids)
    jono.put(asetusarvo.lattia_skaala)
    jono2.put(len(asetusarvo.possibleC))
    for put in asetusarvo.possibleC:
        jono2.put(put)

    #print("tormattavat objektit: ",len(asetusarvo.possibleC)/3)
    teekoo(asetusarvo)
    pygame.mouse.set_visible(False)

    pygame.mixer.music.load("sound/wind.mp3")
    pygame.mixer.music.set_volume(asetusarvo.Mvolume)
    walk = pygame.mixer.Sound("sound/walk2.wav")
    jump = pygame.mixer.Sound("sound/Jump.wav")
    pygame.mixer.Sound.set_volume(walk, asetusarvo.Evolume)
    pygame.mixer.Sound.set_volume(jump, asetusarvo.Evolume)
    pygame.mixer.music.play(-1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    frisbee = kiekko(7, 5, -2, 2)
    frisbee.hyzerkulma = -2
    frisbee.hyzer(frisbee.hyzerkulma)
    frisbee.Kkulma = 2
    if frisbee.Kkulma == 0:
        frisbee.Kkulma = 1
    frisbee.korkeuskulma(frisbee.Kkulma)

    while True:

        ############# PYGAME EVENT HANDLING #############
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sulje.value = 1
                pygame.quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sulje.value = 1
                    pygame.quit()
                    break
                if event.key == pygame.K_d:
                    asetusarvo.XZ_move_ad = +0.6
                if event.key == pygame.K_a:
                    asetusarvo.XZ_move_ad = -0.6
                if event.key == pygame.K_w:
                    asetusarvo.XZ_move_ws = +0.6
                if event.key == pygame.K_s:
                    asetusarvo.XZ_move_ws = -0.6
                if event.key == pygame.K_f:
                    if asetusarvo.luhtu <= 0:
                        asetusarvo.luhtu = asetusarvo.luhtu_strena

                        glUseProgram(perus_shader)
                        glUniform1f(glGetUniformLocation(perus_shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(Ishader)
                        glUniform1f(glGetUniformLocation(Ishader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(I2shader)
                        glUniform1f(glGetUniformLocation(I2shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(T_shader)
                        glUniform1f(glGetUniformLocation(T_shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(T2_shader)
                        glUniform1f(glGetUniformLocation(T2_shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)


                        glUseProgram(Irefshader)
                        glUniform1f(glGetUniformLocation(Irefshader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(I2refshader)
                        glUniform1f(glGetUniformLocation(I2refshader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                    else:
                        asetusarvo.luhtu = 0

                        glUseProgram(perus_shader)
                        glUniform1f(glGetUniformLocation(perus_shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(Ishader)
                        glUniform1f(glGetUniformLocation(Ishader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(I2shader)
                        glUniform1f(glGetUniformLocation(I2shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(T_shader)
                        glUniform1f(glGetUniformLocation(T_shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(T2_shader)
                        glUniform1f(glGetUniformLocation(T2_shader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(Irefshader)
                        glUniform1f(glGetUniformLocation(Irefshader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)

                        glUseProgram(I2refshader)
                        glUniform1f(glGetUniformLocation(I2refshader, "luhtu"), asetusarvo.luhtu)
                        glUseProgram(0)
                    glUseProgram(0)

                if event.key == pygame.K_SPACE:
                    if (asetusarvo.lattiassa_kiinni or asetusarvo.collision == 1) and asetusarvo.deny_jump == 0:
                        asetusarvo.hahmo_koords[1] -= 0.1
                        asetusarvo.Y_vektori = asetusarvo.jump
                        asetusarvo.lattiassa_kiinni = False
                        pygame.mixer.Sound.play(jump)

                if event.key == pygame.K_g:
                    pygame.mouse.set_visible(True)
                    teekoo(asetusarvo)
                    pygame.mixer.music.set_volume(asetusarvo.Mvolume)
                    pygame.mixer.Sound.set_volume(walk, asetusarvo.Evolume)
                    pygame.mixer.Sound.set_volume(jump, asetusarvo.Evolume)
                    pygame.mouse.set_visible(False)

                if event.key == pygame.K_l:
                    x_addon = 1 * math.sin(math.radians(asetusarvo.y_rotation))
                    z_addon = -1 * math.cos(math.radians(asetusarvo.y_rotation))
                    #model = pyrr.Vector3([-asetusarvo.hahmo_koords[0] + x_addon, -asetusarvo.hahmo_koords[1] - 0.5,-asetusarvo.hahmo_koords[2] + z_addon])
                    frisbee.paikkavektori[0] = -asetusarvo.hahmo_koords[0] + x_addon
                    frisbee.paikkavektori[1] = -asetusarvo.hahmo_koords[2] + z_addon
                    frisbee.korkeus = -asetusarvo.hahmo_koords[1] - 0.5
                    frisbee.nopeus_vektori = [x_addon*20,z_addon*20]
                    frisbee.heitto = True


                if event.key == pygame.K_p:
                    pause = True
                    while pause:
                        pygame.mouse.set_visible(True)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    pygame.quit()
                                    return
                                if event.key == pygame.K_p:
                                    pause = False
                        pygame.time.wait(500)
                    pygame.mouse.set_visible(False)

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    asetusarvo.zoom = 0
                if event.key == pygame.K_s:
                    asetusarvo.zoom = 0
                if event.key == pygame.K_a:
                    asetusarvo.XZ_move_ad = 0
                if event.key == pygame.K_d:
                    asetusarvo.XZ_move_ad = 0
                if event.key == pygame.K_w:
                    asetusarvo.XZ_move_ws = 0
                if event.key == pygame.K_s:
                    asetusarvo.XZ_move_ws = 0
        ###################################################


        #movement
        hahmo_move(asetusarvo)
        mouse_freelook(asetusarvo)

        #lattialla pysyminen
        hk0.value = asetusarvo.hahmo_koords[0]
        hk2.value = asetusarvo.hahmo_koords[2]
        lahin_y = hk1.value
        hahmo_Y(asetusarvo, lahin_y)

        #collision detect
        hahmoY.value = asetusarvo.hahmo_koords[1]
        C_type = Ctype.value
        asetusarvo.deny_jump = deny_jump.value
        asetusarvo.collision = collision.value
        collision_handling(asetusarvo, C_type, hcy)


        asetusarvo.Client2pos = [c2x.value,c2y.value,c2z.value]
        #print(asetusarvo.hahmo_koords)

        if asetusarvo.XZ_move_ws + asetusarvo.XZ_move_ad * 2 != 0 and asetusarvo.moving == False and (asetusarvo.deny_jump==False or asetusarvo.lattiassa_kiinni):
            asetusarvo.moving = True
            pygame.mixer.Sound.play(walk, loops=-1)
        if asetusarvo.XZ_move_ws + asetusarvo.XZ_move_ad * 2 == 0 and asetusarvo.moving == True or (-lahin_y+2.7 < -asetusarvo.hahmo_koords[1] and asetusarvo.collision == False):
            asetusarvo.moving = False
            pygame.mixer.Sound.stop(walk)
        #print(ase)

        if frisbee.heitto:
            frisbee.paikka()


        #RENDER
        glEnable(GL_DEPTH_TEST)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_screen)
        glClearColor(0, 0, 0, 0)
        #if asetusarvo.depth_screen:
            # KOKO SCENE TEXTUURIIN ELI FBO HOMMIA
            #glViewport(0, 0, 1024, 1024)
            #shader = Dshader  # match
            #shader_type = "Dshader"  # match
            #render_FBO_depth(fbo_depth, asetusarvo, kolmio, lattia, shader, shader_type, pallo, vesi)


        glViewport(0, 0, 1280, 800)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        shader = no_light_shader  # match
        shader_type = "no_light_shader"  # match
        render_client2(asetusarvo, shader, shader_type, kolmio)
        render_luhtu(asetusarvo,pallo,shader,shader_type,frisbee)

        shader = perus_shader  # match
        shader_type = "perus_shader"  # match
        VP(asetusarvo, shader)
        render(asetusarvo, kolmio, shader, shader_type, pallo)

        #if asetusarvo.reflect == False:
            #render_vesi(asetusarvo, shader, shader_type, vesi)
        glUseProgram(0)


        shader = Ishader  # match
        shader_type = "Ishader"  # match
        render_instanssi(asetusarvo, rock, plant, shader, shader_type, runko,instacube,I2shader,kasvi,pine)



        if asetusarvo.terrain_Mtex_Mnorm:
            shader = T_shader  # match
            shader_type = "T_shader"  # match
        else:
            shader = T2_shader  # match
            shader_type = "T2_shader"  # match
        render_terrain(lattia, asetusarvo, shader, shader_type)



        shader = no_light_shader  # match
        shader_type = "no_light_shader"  # match
        render_nolight(asetusarvo, shader, shader_type, skybox)


        #if asetusarvo.depth_screen:
            #shader = no_light_shader  # match
            #shader_type = "no_light_shader"  # match
            #render_ikkuna(asetusarvo, shader, shader_type, plane)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        if asetusarvo.reflect:
            glViewport(0, 0, 1024, 1024)
            render_FBO_reflect(fbo_reflect, asetusarvo, kolmio, lattia_ref, shader, shader_type, pallo, skybox, perus_shader, Irefshader, rock,plant,runko,instacube,T_shader,T2_shader,kasvi,I2refshader,pine)
            glViewport(0, 0, 1280, 800)
            shader = ref_shader
            shader_type = "ref_shader"
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_screen)
            render_reflect_plane(asetusarvo, shader, shader_type, reflect_plane)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        #Elämä tekstuurissa
        glDisable(GL_DEPTH_TEST)

        if asetusarvo.GLOW == 1:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_brightspot)
            shader = bs_shader  # match
            shader_type = "bs_shader"  # match
            glUseProgram(shader)
            glClear(GL_COLOR_BUFFER_BIT)
            main_screen.piirra(shader, shader_type)
            glUseProgram(0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            glBindFramebuffer(GL_FRAMEBUFFER, fbo_hor_blur_screen)
            glViewport(0, 0, 320, 200)
            shader = hor_blur_shader  # match
            shader_type = "hor_blur_shader"  # match
            glUseProgram(shader)
            glClear(GL_COLOR_BUFFER_BIT)
            main_screen.piirra(shader, shader_type)
            glUseProgram(0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            glBindFramebuffer(GL_FRAMEBUFFER, fbo_ver_blur_screen)
            shader = ver_blur_shader  # match
            shader_type = "ver_blur_shader"  # match
            glUseProgram(shader)
            glClear(GL_COLOR_BUFFER_BIT)
            main_screen.piirra(shader, shader_type)
            glUseProgram(0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        #kuva näytölle
        glViewport(0, 0, 1280, 800)
        shader = F_shader  # match
        shader_type = "F_shader"  # match
        glUseProgram(shader)
        glUniform1i(glGetUniformLocation(shader, "postprocess"), asetusarvo.PP)
        glUniform1i(glGetUniformLocation(shader, "motionblur"), asetusarvo.MB)
        glUniform1i(glGetUniformLocation(shader, "glow"), asetusarvo.GLOW)
        glClear(GL_COLOR_BUFFER_BIT)
        main_screen.piirra(shader, shader_type)
        glUseProgram(0)



        pygame.display.flip()

        if asetusarvo.MB == 1:
            #flipin jalkeen seuraavalle rundille edellisen kuva
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_prevframe1)
            shader = F2_shader  # match
            shader_type = "F2_shader"  # match
            glUseProgram(shader)
            glUniform1i(glGetUniformLocation(shader, "postprocess"), asetusarvo.PP)
            glUniform1i(glGetUniformLocation(shader, "glow"), asetusarvo.GLOW)
            glClear(GL_COLOR_BUFFER_BIT)
            main_screen.piirra(shader, shader_type)
            glUseProgram(0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)
        #asetusarvo.SL = math.sin(pygame.time.get_ticks()*0.0001)+2-1
        pygame.time.wait(1)
        #Clock.tick(75)




########## igguna ja init shaissee#####
def init_window(asetusarvo):
    pygame.init()
    pygame.display.set_mode(asetusarvo.display, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption('heneEngine')
    kulli = pygame.image.load("ikoni.jpg")
    pygame.display.set_icon(kulli)
    pygame.mixer.init()
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_CLIP_DISTANCE0)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # glEnable(GL_ALPHA_TEST)
    # glBlendFunc(GL_ONE, GL_ONE)

#todella epaonnistuneet loadingscreenit
def loading_shaiba():
    vertex_shader = """
                        #version 330
                        in vec3 position;
                        in vec3 color;

                        out vec3 newColor;

                        void main()
                        {
                            gl_Position = vec4(position,1.0f);
                            newColor = color;
                        }
                        """

    fragment_shader = """
                        #version 330
                        in vec3 newColor;


                        out vec4 outColor;
                        void main()
                        {
                            outColor = vec4(newColor, 1.0f);
                        }
                        """

    loading_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))

    loading_model = [-0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                    0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.5, 0.0, 0.0, 0.0, 1.0]

    loading_model = numpy.array(loading_model, dtype=numpy.float32)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 72, loading_model, GL_STATIC_DRAW)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    position = glGetAttribLocation(loading_shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(loading_shader, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)





    return loading_shader ,VAO
def draw_loading(loader_shader):
    glUseProgram(loader_shader[0])
    glClearColor(random.random()/4,random.random()/6,random.random()/6,1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindVertexArray(loader_shader[1])
    glDrawArrays(GL_TRIANGLES,0,3)
    pygame.display.flip()
    glUseProgram(0)
    glBindVertexArray(0)


########## rendershaissee #############
def VP(asetusarvo, shader):
    glUseProgram(shader)

    # koko maailman liikuttelu aka hahmon liikkuminen
    view_trans = pyrr.matrix44.create_from_translation(
        [asetusarvo.hahmo_koords[0], asetusarvo.hahmo_koords[1], asetusarvo.hahmo_koords[2]])
    rot_xyz = [-asetusarvo.x_rotation, -asetusarvo.y_rotation, 0]
    view_rot = rotate_XYZ(rot_xyz)
    view = view_rot * view_trans
    view_loc = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # lightpos = pyrr.matrix44.create_from_translation([0,0,3])
    lightpos = pyrr.matrix44.create_from_translation(
        pyrr.Vector3(
            [-asetusarvo.hahmo_koords[0] - 1.2, -asetusarvo.hahmo_koords[1] - 0.5, -asetusarvo.hahmo_koords[2] - 1.2]))
    Lpos = glGetUniformLocation(shader, "lightpos")
    glUniformMatrix4fv(Lpos, 1, GL_FALSE, lightpos)

    asetusarvo.Lpos = lightpos

    viewPos = pyrr.matrix44.create_from_translation(
        pyrr.Vector3([-asetusarvo.hahmo_koords[0], -asetusarvo.hahmo_koords[1], -asetusarvo.hahmo_koords[2]]))
    Vpos = glGetUniformLocation(shader, "viewPos")
    glUniformMatrix4fv(Vpos, 1, GL_FALSE, viewPos)

    # tan muuttaminen... EI
    projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1280 / 800, 0.1, 3000.0)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    asetusarvo.P = projection
    asetusarvo.V = view
    asetusarvo.Lpos = lightpos
    asetusarvo.Vpos = viewPos
def scale_object(obj_scale):
    scale = numpy.array([[obj_scale, 0.0, 0.0, 0.0],
                         [0.0, obj_scale, 0.0, 0.0],
                         [0.0, 0.0, obj_scale, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=numpy.float32)
    return scale
def rotate_XYZ(rot_xyz):
    x = (math.radians(rot_xyz[0]))
    y = (math.radians(rot_xyz[1]))
    z = (math.radians(rot_xyz[2]))
    rot_x = pyrr.Matrix44.from_x_rotation(x)
    rot_y = pyrr.Matrix44.from_y_rotation(y)
    rot_z = pyrr.Matrix44.from_z_rotation(z)
    rot = rot_x * rot_y * rot_z
    return rot


######## normirender ########
def render(asetusarvo, kolmio, shader, shader_type, pallo):

    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 2.0)
    glUniform1f(glGetUniformLocation(shader, "ambientSTR"), asetusarvo.AL)
    glUniform1f(glGetUniformLocation(shader, "sunlightSTR"), asetusarvo.SL)


    # cube1
    model = asetusarvo.cube_pos_list[0]
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [pygame.time.get_ticks() / 50, pygame.time.get_ticks() / 100, pygame.time.get_ticks() / 100]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.01
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    kolmio.piirra(shader,shader_type)

    # cube2
    model = asetusarvo.cube_pos_list[1]
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.01
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    kolmio.piirra(shader,shader_type)

    # cube3
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 10.0)
    model = asetusarvo.cube_pos_list[2]
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [pygame.time.get_ticks() / 10, pygame.time.get_ticks() / 15, pygame.time.get_ticks() / 20]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.01
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    kolmio.piirra(shader,shader_type)

    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 1.0)
    # cube4
    model = asetusarvo.cube_pos_list[3]
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [pygame.time.get_ticks() / 100, pygame.time.get_ticks() / 100, pygame.time.get_ticks() / 50]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.01
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    kolmio.piirra(shader,shader_type)


    # boool
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 10.0)
    model = asetusarvo.pallo_pos
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, pygame.time.get_ticks() / 10, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.1
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    pallo.piirra(shader,shader_type)
def render_terrain(lattia,asetusarvo,shader,shader_type):
    # lattia
    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(shader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 0.1)
    glUniform1f(glGetUniformLocation(shader, "ambientSTR"), asetusarvo.AL)
    glUniform1f(glGetUniformLocation(shader, "sunlightSTR"), asetusarvo.SL)

    model = asetusarvo.lattia_pos
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = asetusarvo.lattia_skaala
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    lattia.piirra(shader, shader_type)
    glUseProgram(0)
def render_instanssi(asetusarvo, rock, plant, shader, shader_type, runko,instacube,I2shader,kasvi,pine):
    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(shader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    glUniform1f(glGetUniformLocation(shader, "ambientSTR"), asetusarvo.AL)
    glUniform1f(glGetUniformLocation(shader, "sunlightSTR"), asetusarvo.SL)

    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 2.0)
    rock.piirra(shader,shader_type)

    runko.piirra(shader,shader_type)

    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 5.0)
    instacube.piirra(shader,shader_type)


    glUseProgram(0)


    glUseProgram(I2shader)

    glUniformMatrix4fv(glGetUniformLocation(I2shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(I2shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(I2shader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(I2shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    glUniform1f(glGetUniformLocation(I2shader, "sunlightSTR"), asetusarvo.SL)
    glUniform1f(glGetUniformLocation(I2shader, "time"), (pygame.time.get_ticks()/100))

    glUniform1f(glGetUniformLocation(I2shader, "specularStrenght"), 2.0)
    glUniform1f(glGetUniformLocation(I2shader, "Voima"), 1.0)
    glUniform1f(glGetUniformLocation(I2shader, "ambientSTR"), asetusarvo.AL + asetusarvo.SL*2)
    plant.piirra(I2shader, shader_type)
    if asetusarvo.kasvi:
        glUniform1f(glGetUniformLocation(I2shader, "ambientSTR"), asetusarvo.AL + asetusarvo.SL)
        glUniform1f(glGetUniformLocation(I2shader, "specularStrenght"), 0.0)
        glUniform1f(glGetUniformLocation(I2shader, "Voima"), 0.1)
        kasvi.piirra(I2shader, shader_type)
        glUniform1f(glGetUniformLocation(I2shader, "Voima"), 0.05)
        pine.piirra(I2shader, shader_type)
    glUseProgram(0)


def render_nolight(asetusarvo, shader, shader_type, skybox):

    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniform1f(glGetUniformLocation(shader, "sunStr"), asetusarvo.SL)

    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, pygame.time.get_ticks() / 500,0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 30*asetusarvo.lattia_skaala
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    skybox.piirra(shader,shader_type)
    glUseProgram(0)
def render_vesi(asetusarvo, shader, shader_type, vesi):

    glEnable(GL_BLEND)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = asetusarvo.lattia_skaala
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    vesi.piirra(shader, shader_type)
    glDisable(GL_BLEND)
def render_luhtu(asetusarvo,pallo,shader,shader_type, frisbee):
    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 2.0)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")

    #x_addon = 1 * math.sin(math.radians(asetusarvo.y_rotation))
    #z_addon = -1 * math.cos(math.radians(asetusarvo.y_rotation))

    #model = pyrr.matrix44.create_from_translation(pyrr.Vector3(
        #[-asetusarvo.hahmo_koords[0] + x_addon, -asetusarvo.hahmo_koords[1] - 0.5,
         #-asetusarvo.hahmo_koords[2] + z_addon]))
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(
        [frisbee.paikkavektori[0], frisbee.korkeus,
         frisbee.paikkavektori[1]]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, asetusarvo.y_rotation, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.02
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    pallo.piirra(shader, shader_type)
    glUseProgram(0)

def render_client2(asetusarvo, shader, shader_type, kolmio):
    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")

    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(asetusarvo.Client2pos))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, pygame.time.get_ticks() / 500, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.01
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    kolmio.piirra(shader, shader_type)
    glUseProgram(0)


############ movement ###############
def mouse_freelook(asetusarvo):
    mousepos_x = pygame.mouse.get_pos()[0]
    asetusarvo.y_rotation -= ((asetusarvo.mousepos_edellinen_x - mousepos_x) * asetusarvo.sens)
    asetusarvo.mousepos_edellinen_x = 640

    edellinen_x_rot = asetusarvo.x_rotation

    mousepos_y = pygame.mouse.get_pos()[1]
    if asetusarvo.invert_y:
        asetusarvo.x_rotation -= ((asetusarvo.mousepos_edellinen_y - mousepos_y) * asetusarvo.sens)
    else:
        asetusarvo.x_rotation += ((asetusarvo.mousepos_edellinen_y - mousepos_y) * asetusarvo.sens)
    asetusarvo.mousepos_edellinen_y = 400
    pygame.mouse.set_pos(640, 400)

    tarkastelu = (math.sin(math.radians(asetusarvo.x_rotation)))
    if tarkastelu >= 0.98 or tarkastelu <= -0.98:
        asetusarvo.x_rotation = edellinen_x_rot
def hahmo_move(asetusarvo):
    asetusarvo.XZ_MOVE_ad += asetusarvo.XZ_move_ad
    asetusarvo.XZ_MOVE_ws += asetusarvo.XZ_move_ws
    if asetusarvo.XZ_move_ws == 0:
        asetusarvo.XZ_MOVE_ws = 0
    if asetusarvo.XZ_move_ad == 0:
        asetusarvo.XZ_MOVE_ad = 0

    if asetusarvo.XZ_MOVE_ws >= asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ws = asetusarvo.Max_nopeus
    if asetusarvo.XZ_MOVE_ws <= -asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ws = -asetusarvo.Max_nopeus

    if asetusarvo.XZ_MOVE_ad >= asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ad = asetusarvo.Max_nopeus
    if asetusarvo.XZ_MOVE_ad <= -asetusarvo.Max_nopeus:
        asetusarvo.XZ_MOVE_ad = -asetusarvo.Max_nopeus

    x_kerroin_ws = (asetusarvo.Kiihtyvyys) * math.sin(math.radians(asetusarvo.y_rotation))
    z_kerroin_ws = -(asetusarvo.Kiihtyvyys) * math.cos(math.radians(asetusarvo.y_rotation))

    x_kerroin_ad = (asetusarvo.Kiihtyvyys) * math.sin(math.radians(asetusarvo.y_rotation + 90))
    z_kerroin_ad = -(asetusarvo.Kiihtyvyys) * math.cos(math.radians(asetusarvo.y_rotation + 90))

    asetusarvo.XZ_move_vektori = [-(asetusarvo.XZ_MOVE_ws * x_kerroin_ws) - (asetusarvo.XZ_MOVE_ad * x_kerroin_ad),
                                  -(asetusarvo.XZ_MOVE_ws * z_kerroin_ws) - (asetusarvo.XZ_MOVE_ad * z_kerroin_ad)]
    asetusarvo.hahmo_koords[0] += asetusarvo.XZ_move_vektori[0]
    asetusarvo.hahmo_koords[2] += asetusarvo.XZ_move_vektori[1]

###############FBOdephtscreen###################
def FBO_depth(plane):
    depthMapFBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, plane.img_index, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return depthMapFBO
def render_FBO_depth(fbo_depth, asetusarvo, kolmio, lattia, shader, shader_type, pallo, vesi):

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_depth)
    #glClearColor(0, 0, 0, 0)
    glClear(GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    render(asetusarvo, kolmio, lattia, shader, shader_type, pallo)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glUseProgram(0)
def render_ikkuna(asetusarvo, shader, shader_type, plane):
    glUseProgram(shader)

    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)

    # plane
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([5, 7, 9]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 150, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.1
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    plane.piirra(shader,shader_type)
    glUseProgram(0)

######### FBO reflect #####
def FBO_reflect(reflect_plane):
    reflect_buff = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, reflect_buff)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 1024)

    FBO_reflect = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO_reflect)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, reflect_plane.img_index, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, reflect_buff)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return FBO_reflect
def render_FBO_reflect(fbo_reflect, asetusarvo, kolmio, lattia_ref, shader, shader_type, pallo, skybox, perus_shader, Irefshader,rock,plant,runko,instacube,T_shader,T2_shader,kasvi,I2refshader,pine):

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_reflect)
    glClearColor(0, 0, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader)

    view_trans = pyrr.matrix44.create_from_translation(
        [asetusarvo.hahmo_koords[0], -asetusarvo.hahmo_koords[1], asetusarvo.hahmo_koords[2]])
    rot_xyz = [asetusarvo.x_rotation, -asetusarvo.y_rotation, 0]
    view_rot = rotate_XYZ(rot_xyz)
    view = view_rot * view_trans

    asetusarvo.Vref = view

    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniform1f(glGetUniformLocation(shader, "sunStr"), asetusarvo.SL)

    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, pygame.time.get_ticks() / 500, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 30*asetusarvo.lattia_skaala
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    skybox.piirra(shader, shader_type)


    glUseProgram(0)

    shader = perus_shader
    shader_type = "perus_shader"
    glUseProgram(shader)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.Vref)
    glUniformMatrix4fv(glGetUniformLocation(shader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    render(asetusarvo, kolmio, shader, shader_type, pallo)
    glUseProgram(0)

    if asetusarvo.terrain_Mtex_Mnorm:
        shader = T_shader  # match
        shader_type = "T_shader"  # match
    else:
        shader = T2_shader  # match
        shader_type = "T2_shader"  # match
    render_terrain_ref(lattia_ref, asetusarvo, shader, shader_type)

    shader = Irefshader
    shader_type = "Ishader"
    render_instanssi_ref(asetusarvo, rock, plant, shader, shader_type, runko, instacube, kasvi, I2refshader, pine)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
def render_reflect_plane(asetusarvo, shader, shader_type, reflect_plane):
    glUseProgram(shader)

    asetusarvo.disto_move += asetusarvo.disto_move_speed*0.1

    glUniform1f(glGetUniformLocation(shader, "move"), asetusarvo.disto_move)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniform1f(glGetUniformLocation(shader, "sunStr"), 1000)

    # plane
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = asetusarvo.lattia_skaala
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    reflect_plane.piirra(shader,shader_type)
    glUseProgram(0)
def render_instanssi_ref(asetusarvo, rock, plant, shader, shader_type, runko,instacube,kasvi,I2refshader, pine):
    glUseProgram(shader)


    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.Vref)
    glUniformMatrix4fv(glGetUniformLocation(shader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    glUniform1f(glGetUniformLocation(shader, "ambientSTR"), asetusarvo.AL)
    glUniform1f(glGetUniformLocation(shader, "sunlightSTR"), asetusarvo.SL)

    rock.piirra(shader,shader_type)
    runko.piirra(shader,shader_type)
    instacube.piirra(shader, shader_type)

    glUseProgram(0)
    glUseProgram(I2refshader)

    glUniformMatrix4fv(glGetUniformLocation(I2refshader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(I2refshader, "view"), 1, GL_FALSE, asetusarvo.Vref)
    glUniformMatrix4fv(glGetUniformLocation(I2refshader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(I2refshader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    glUniform1f(glGetUniformLocation(I2refshader, "sunlightSTR"), asetusarvo.SL)
    glUniform1f(glGetUniformLocation(I2refshader, "time"), (pygame.time.get_ticks() / 200))
    glUniform1f(glGetUniformLocation(I2refshader, "ambientSTR"), asetusarvo.AL + asetusarvo.SL)

    glUniform1f(glGetUniformLocation(I2refshader, "Voima"), 1.0)
    plant.piirra(I2refshader, shader_type)
    if asetusarvo.kasvi:
        glUniform1f(glGetUniformLocation(I2refshader, "Voima"), 0.1)
        kasvi.piirra(I2refshader, shader_type)
        glUniform1f(glGetUniformLocation(I2refshader, "Voima"), 0.05)
        pine.piirra(I2refshader, shader_type)

    glUseProgram(0)
def render_terrain_ref(lattia_ref,asetusarvo,shader,shader_type):
    # lattia
    glUseProgram(shader)

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.Vref)
    glUniformMatrix4fv(glGetUniformLocation(shader, "viewPos"), 1, GL_FALSE, asetusarvo.Vpos)
    glUniformMatrix4fv(glGetUniformLocation(shader, "lightpos"), 1, GL_FALSE, asetusarvo.Lpos)
    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniform1f(glGetUniformLocation(shader, "specularStrenght"), 0.1)
    glUniform1f(glGetUniformLocation(shader, "ambientSTR"), asetusarvo.AL)
    glUniform1f(glGetUniformLocation(shader, "sunlightSTR"), asetusarvo.SL)

    model = asetusarvo.lattia_pos
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 0, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = asetusarvo.lattia_skaala
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    lattia_ref.piirra(shader, shader_type)
    glUseProgram(0)

###### FBO ikkuna #####
def FBO_plane2_render(plane2):
    brightspot = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, brightspot)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1280, 800)

    FBO_BS = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO_BS)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, plane2.img_index, 0) #imgindex
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, brightspot)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return FBO_BS
def render_ikkuna2(asetusarvo, shader, shader_type, plane2):
    glUseProgram(shader)

    transform_loc = glGetUniformLocation(shader, "transform")
    model_loc = glGetUniformLocation(shader, "model")
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, asetusarvo.V)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, asetusarvo.P)

    # plane
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([5, 7, 9]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    rot_xyz = [0, 150, 0]
    rot = rotate_XYZ(rot_xyz)
    obj_scale = 0.1
    scale = scale_object(obj_scale)
    transform = rot * scale
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
    plane2.piirra(shader,shader_type)
    glUseProgram(0)

### postprocessFBOs ####
def FBOscreen():
    SCRbuf = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, SCRbuf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1280, 800)

    fboscreen = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fboscreen)

    screen = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, screen)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 800, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screen, 0) #imgindex
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, SCRbuf)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("screen",screen)
    return fboscreen
def FBO_hor_blur_screen():
    SCR_process_buf = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, SCR_process_buf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 320, 200)

    fbo_process_screen = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_process_screen)

    process_screen = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, process_screen)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 320, 200, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, process_screen, 0)  # imgindex
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, SCR_process_buf)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("hor_blur_buffer",process_screen)
    return fbo_process_screen
def FBO_ver_blur_screen():
    SCR_process_buf = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, SCR_process_buf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 320, 200)

    fbo_process_screen = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_process_screen)

    process_screen = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, process_screen)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 320, 200, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, process_screen, 0)  # imgindex
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, SCR_process_buf)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("ver_blur_buffer",process_screen)
    return fbo_process_screen
def FBO_brightspot():
    SCRbuf = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, SCRbuf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1280, 800)

    fboscreen = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fboscreen)

    screen = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, screen)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 800, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screen, 0)  # imgindex
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, SCRbuf)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("brightspot_buffer",screen)
    return fboscreen
def FBO_prevframe1():
    SCRbuf = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, SCRbuf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1280, 800)

    fboscreen = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fboscreen)

    screen = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, screen)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 800, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screen, 0)  # imgindex
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, SCRbuf)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("prevframe1", screen)
    return fboscreen

##### lattia grideiks
def lattia_grid(asetusarvo, lattia_lista):
    for x in range(len(lattia_lista)):
        grid_1 = []
        grid_2 = []
        grid_3 = []
        grid_4 = []
        grid_5 = []
        grid_6 = []
        grid_7 = []
        grid_8 = []
        grid_9 = []
        grid_10 = []
        grid_11 = []
        grid_12 = []
        grid_13 = []
        grid_14 = []
        grid_15 = []
        grid_16 = []

        grid_17 = []
        grid_18 = []
        grid_19 = []
        grid_20 = []
        grid_21 = []
        grid_22 = []
        grid_23 = []
        grid_24 = []
        grid_25 = []
        grid_26 = []
        grid_27 = []
        grid_28 = []
        grid_29 = []
        grid_30 = []
        grid_31 = []
        grid_32 = []

        grid_33 = []
        grid_34 = []
        grid_35 = []
        grid_36 = []
        grid_37 = []
        grid_38 = []
        grid_39 = []
        grid_40 = []
        grid_41 = []
        grid_42 = []
        grid_43 = []
        grid_44 = []
        grid_45 = []
        grid_46 = []
        grid_47 = []
        grid_48 = []

        grid_49 = []
        grid_50 = []
        grid_51 = []
        grid_52 = []
        grid_53 = []
        grid_54 = []
        grid_55 = []
        grid_56 = []
        grid_57 = []
        grid_58 = []
        grid_59 = []
        grid_60 = []
        grid_61 = []
        grid_62 = []
        grid_63 = []
        grid_64 = []

        #for laskin in range(0,len(lattia_lista[x].vertexnormal,2)):
        for osio in lattia_lista[x].verticies:
            pygame.event.get()
            # + +
            if osio[0] >= 0 and osio[2] >= 0:
                if osio[0] >= 0 and osio[0] <= 125 and osio[2] >= 0 and osio[2] <= 125:
                    grid_1.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] >= 0 and osio[2] <= 125:
                    grid_2.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] >= 0 and osio[2] <= 125:
                    grid_3.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] >= 0 and osio[2] <= 125:
                    grid_4.append(osio)

                elif osio[0] >= 0 and osio[0] <= 125 and osio[2] >= 125 and osio[2] <= 250:
                    grid_5.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] >= 125 and osio[2] <= 250:
                    grid_6.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] >= 125 and osio[2] <= 250:
                    grid_7.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] >= 125 and osio[2] <= 250:
                    grid_8.append(osio)

                elif osio[0] >= 0 and osio[0] <= 125 and osio[2] >= 250 and osio[2] <= 375:
                    grid_9.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] >= 250 and osio[2] <= 375:
                    grid_10.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] >= 250 and osio[2] <= 375:
                    grid_11.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] >= 250 and osio[2] <= 375:
                    grid_12.append(osio)

                elif osio[0] >= 0 and osio[0] <= 125 and osio[2] >= 375 and osio[2] <= 500:
                    grid_13.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] >= 375 and osio[2] <= 500:
                    grid_14.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] >= 375 and osio[2] <= 500:
                    grid_15.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] >= 375 and osio[2] <= 500:
                    grid_16.append(osio)

            # + -
            elif osio[0] >= 0 and osio[2] <= 0:
                if osio[0] >= 0 and osio[0] <= 125 and osio[2] <= 0 and osio[2] >= -125:
                    grid_17.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] <= 0 and osio[2] >= -125:
                    grid_18.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] <= 0 and osio[2] >= -125:
                    grid_19.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] <= 0 and osio[2] >= -125:
                    grid_20.append(osio)

                elif osio[0] >= 0 and osio[0] <= 125 and osio[2] <= -125 and osio[2] >= -250:
                    grid_21.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] <= -125 and osio[2] >= -250:
                    grid_22.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] <= -125 and osio[2] >= -250:
                    grid_23.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] <= -125 and osio[2] >= -250:
                    grid_24.append(osio)

                elif osio[0] >= 0 and osio[0] <= 125 and osio[2] <= -250 and osio[2] >= -375:
                    grid_25.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] <= -250 and osio[2] >= -375:
                    grid_26.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] <= -250 and osio[2] >= -375:
                    grid_27.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] <= -250 and osio[2] >= -375:
                    grid_28.append(osio)

                elif osio[0] >= 0 and osio[0] <= 125 and osio[2] <= -375 and osio[2] >= -500:
                    grid_29.append(osio)
                elif osio[0] >= 125 and osio[0] <= 250 and osio[2] <= -375 and osio[2] >= -500:
                    grid_30.append(osio)
                elif osio[0] >= 250 and osio[0] <= 375 and osio[2] <= -375 and osio[2] >= -500:
                    grid_31.append(osio)
                elif osio[0] >= 375 and osio[0] <= 500 and osio[2] <= -375 and osio[2] >= -500:
                    grid_32.append(osio)

            # - -
            elif osio[0] <= 0 and osio[2] <= 0:
                if osio[0] <= 0 and osio[0] >= -125 and osio[2] <= 0 and osio[2] >= -125:
                    grid_33.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] <= 0 and osio[2] >= -125:
                    grid_34.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] <= 0 and osio[2] >= -125:
                    grid_35.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] <= 0 and osio[2] >= -125:
                    grid_36.append(osio)

                elif osio[0] <= 0 and osio[0] >= -125 and osio[2] <= -125 and osio[2] >= -250:
                    grid_37.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] <= -125 and osio[2] >= -250:
                    grid_38.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] <= -125 and osio[2] >= -250:
                    grid_39.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] <= -125 and osio[2] >= -250:
                    grid_40.append(osio)

                elif osio[0] <= 0 and osio[0] >= -125 and osio[2] <= -250 and osio[2] >= -375:
                    grid_41.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] <= -250 and osio[2] >= -375:
                    grid_42.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] <= -250 and osio[2] >= -375:
                    grid_43.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] <= -250 and osio[2] >= -375:
                    grid_44.append(osio)

                elif osio[0] <= 0 and osio[0] >= -125 and osio[2] <= -375 and osio[2] >= -500:
                    grid_45.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] <= -375 and osio[2] >= -500:
                    grid_46.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] <= -375 and osio[2] >= -500:
                    grid_47.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] <= -375 and osio[2] >= -500:
                    grid_48.append(osio)

            # - +
            elif osio[0] <= 0 and osio[2] >= 0:
                if osio[0] <= 0 and osio[0] >= -125 and osio[2] >= 0 and osio[2] <= 125:
                    grid_49.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] >= 0 and osio[2] <= 125:
                    grid_50.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] >= 0 and osio[2] <= 125:
                    grid_51.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] >= 0 and osio[2] <= 125:
                    grid_52.append(osio)

                elif osio[0] <= 0 and osio[0] >= -125 and osio[2] >= 125 and osio[2] <= 250:
                    grid_53.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] >= 125 and osio[2] <= 250:
                    grid_54.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] >= 125 and osio[2] <= 250:
                    grid_55.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] >= 125 and osio[2] <= 250:
                    grid_56.append(osio)

                elif osio[0] <= 0 and osio[0] >= -125 and osio[2] >= 250 and osio[2] <= 375:
                    grid_57.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] >= 250 and osio[2] <= 375:
                    grid_58.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] >= 250 and osio[2] <= 375:
                    grid_59.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] >= 250 and osio[2] <= 375:
                    grid_60.append(osio)

                elif osio[0] <= 0 and osio[0] >= -125 and osio[2] >= 375 and osio[2] <= 500:
                    grid_61.append(osio)
                elif osio[0] <= -125 and osio[0] >= -250 and osio[2] >= 375 and osio[2] <= 500:
                    grid_62.append(osio)
                elif osio[0] <= -250 and osio[0] >= -375 and osio[2] >= 375 and osio[2] <= 500:
                    grid_63.append(osio)
                elif osio[0] <= -375 and osio[0] >= -500 and osio[2] >= 375 and osio[2] <= 500:
                    grid_64.append(osio)

        asetusarvo.lattia_grids[x] = [grid_1, grid_2, grid_3, grid_4, grid_5, grid_6, grid_7, grid_8, grid_9, grid_10, \
                                      grid_11, grid_12, grid_13, grid_14, grid_15, grid_16, grid_17, grid_18, grid_19, \
                                      grid_20, grid_21, grid_22, grid_23, grid_24, grid_25, grid_26, grid_27, grid_28, \
                                      grid_29, grid_30, grid_31, grid_32, grid_33, grid_34, grid_35, grid_36, grid_37, \
                                      grid_38, grid_39, grid_40, grid_41, grid_42, grid_43, grid_44, grid_45, grid_46, \
                                      grid_47, grid_48, grid_49, grid_50, grid_51, grid_52, grid_53, grid_54, grid_55, \
                                      grid_56, grid_57, grid_58, grid_59, grid_60, grid_61, grid_62, grid_63, grid_64]

###### instanssi random Matrix generaattorit ####
def instance_matrix(lattia, asetusarvo, divisor, torm, nosto,instamap,mapcolor,random_nosto,smart_nosto):
    im = Image.open(instamap)
    px = im.load()
    sivun_pituus = 1000
    instanssi_mat_list = []
    for x in range(len(lattia.verticies)):
        pygame.event.get()
        if x % divisor == 0:
            pixel = [round(lattia.verticies[x][0]), round(lattia.verticies[x][2])]
            pixel[0] += 500
            pixel[1] += 500
            if pixel[0] > sivun_pituus - 1:
                pixel[0] = sivun_pituus - 1
            if pixel[1] > sivun_pituus - 1:
                pixel[1] = sivun_pituus - 1
            #print(pixel[0], pixel[1])
            try:
                color = (px[pixel[0], pixel[1]])
            except:
                color = 0
                pass
            #print(color)

            try:
                if color[0] == mapcolor:
                    if random_nosto:
                        r_nosto = random.random()*nosto*2
                    else:
                        r_nosto = 0

                    if smart_nosto:
                        nosto = nosto
                    else:
                        nosto = nosto
                    pre = [lattia.verticies[x][0] * \
                           asetusarvo.lattia_skaala,
                           (lattia.verticies[x][1] * asetusarvo.lattia_skaala + random.random() / 3) + 0.2+nosto+r_nosto,
                           lattia.verticies[x][2] * \
                           asetusarvo.lattia_skaala + random.random() / 3]
                    mod = pyrr.matrix44.create_from_translation(pyrr.Vector3(pre))
                    if torm[0] == True:
                        asetusarvo.possibleC.append(pre)
                        asetusarvo.possibleC.append(torm[1])
                        asetusarvo.possibleC.append(torm[2])
                    instanssi_mat_list.append(mod)
            except:
                pass
    return instanssi_mat_list
def transform_instanssi(instanssi_mat_list, skaal,random_rot,random_skaal):
    instanssi_trans_list = []
    for x in range(len(instanssi_mat_list)):
        pygame.event.get()
        if random_rot:
            r = random.random() * 360
        else:
            r=0
        rot_xyz = [0, r, 0]
        rot = rotate_XYZ(rot_xyz)
        if random_skaal:
            obj_scale = skaal + skaal * random.random() * 3
        else:
            obj_scale = skaal
        scale = scale_object(obj_scale)
        transform = rot * scale
        instanssi_trans_list.append(transform)
    return instanssi_trans_list
def instanssi_saadot(lattia,asetusarvo,loading_shader,rock,plant,runko,instacube,kasvi,pine):
    torm = [False]  # voko tormata, säde ja voiko menna paalle/alle
    divisor = 127
    skaal = 0.01
    random_rot = True
    random_skaal = True
    smart_nosto = False
    nosto = -0.2
    instamap = "terrain/Itree3.png"
    mapcolor = 255
    random_nosto = False
    instanssi_mat_list = instance_matrix(lattia, asetusarvo, divisor, torm, nosto,instamap,mapcolor,random_nosto,smart_nosto)
    instanssi_trans_list = transform_instanssi(instanssi_mat_list, skaal, random_rot,random_skaal)
    rock.instanceBuffer(instanssi_mat_list, instanssi_trans_list)
    draw_loading(loading_shader)
    print("kivia: ", len(instanssi_mat_list))
    rock_len = len(instanssi_mat_list)

    torm = [False]
    divisor = 3
    skaal = 0.028
    instamap = "terrain/Itree3.png"
    instanssi_mat_list = instance_matrix(lattia, asetusarvo, divisor, torm, nosto,instamap,mapcolor,random_nosto,smart_nosto)
    instanssi_trans_list = transform_instanssi(instanssi_mat_list, skaal, random_rot,random_skaal)
    plant.instanceBuffer(instanssi_mat_list, instanssi_trans_list)
    draw_loading(loading_shader)
    print("pusikkoja: ", len(instanssi_mat_list))
    plant_len = len(instanssi_mat_list)

    torm = [True, 0.5, 0]
    divisor = 259
    skaal = 0.1
    instanssi_mat_list = instance_matrix(lattia, asetusarvo, divisor, torm, nosto,instamap,mapcolor,random_nosto,smart_nosto)
    instanssi_trans_list = transform_instanssi(instanssi_mat_list, skaal, random_rot,random_skaal)
    runko.instanceBuffer(instanssi_mat_list, instanssi_trans_list)
    print("runkoja: ", len(instanssi_mat_list))
    runko_len = len(instanssi_mat_list)

    torm = [False]
    divisor = 79
    skaal = 3
    nosto = 0.3
    instamap = "terrain/Itree3.png"
    instanssi_mat_list = instance_matrix(lattia, asetusarvo, divisor, torm, nosto, instamap, mapcolor,random_nosto,smart_nosto)
    instanssi_trans_list = transform_instanssi(instanssi_mat_list, skaal, random_rot, random_skaal)
    kasvi.instanceBuffer(instanssi_mat_list, instanssi_trans_list)
    draw_loading(loading_shader)
    print("kasveja: ", len(instanssi_mat_list))
    kasvi_len = len(instanssi_mat_list)

    torm = [True, 0.5, 0]
    random_rot = True
    divisor = 13
    skaal = 6
    nosto = -2
    instamap = "terrain/Itree3.png"
    instanssi_mat_list = instance_matrix(lattia, asetusarvo, divisor, torm, nosto, instamap, mapcolor, random_nosto,smart_nosto)
    instanssi_trans_list = transform_instanssi(instanssi_mat_list, skaal, random_rot, random_skaal)
    pine.instanceBuffer(instanssi_mat_list, instanssi_trans_list)
    draw_loading(loading_shader)
    print("kuusia: ", len(instanssi_mat_list))
    pine_len = len(instanssi_mat_list)

    torm = [True, 1.0, 1]
    divisor = 97
    skaal = 0.01
    random_skaal = False
    smart_nosto = False
    nosto = 5
    random_nosto = True
    instanssi_mat_list = instance_matrix(lattia, asetusarvo, divisor, torm, nosto,instamap,mapcolor,random_nosto,smart_nosto)
    instanssi_trans_list = transform_instanssi(instanssi_mat_list, skaal, random_rot,random_skaal)
    instacube.instanceBuffer(instanssi_mat_list, instanssi_trans_list)
    print("kuutioita: ", len(instanssi_mat_list))
    cube_len = len(instanssi_mat_list)

    return [rock_len,plant_len,runko_len,kasvi_len,cube_len,pine_len]

# collision
def hahmo_Y(asetusarvo, lahin_y):
    if asetusarvo.hahmo_koords[1] >= -2.5 and lahin_y > 0:
        asetusarvo.hahmo_koords[1] = -2.5
        asetusarvo.lattiassa_kiinni = True

    elif asetusarvo.hahmo_koords[1] >= lahin_y - 2.5:
        asetusarvo.Y_vektori = (lahin_y-2.5 - asetusarvo.hahmo_koords[1]) / 10
        asetusarvo.hahmo_koords[1] += asetusarvo.Y_vektori
        asetusarvo.lattiassa_kiinni = True
    else:
        asetusarvo.Y_vektori -= asetusarvo.Y_kiihtyvyys
        asetusarvo.hahmo_koords[1] -= asetusarvo.Y_vektori
        asetusarvo.lattiassa_kiinni = False
def collision_handling(asetusarvo,C_type,hcy):
        #ppaalla
    if C_type == 1:
        asetusarvo.hahmo_koords[1] = hcy.value
        asetusarvo.Y_vektori += asetusarvo.Y_kiihtyvyys
        #alla
    if C_type == 2:
        asetusarvo.hahmo_koords[1] = hcy.value
        asetusarvo.Y_vektori = 0
        #sivusta
    if C_type == 3:
        asetusarvo.hahmo_koords[0] -= asetusarvo.XZ_move_vektori[0]
        asetusarvo.hahmo_koords[2] -= asetusarvo.XZ_move_vektori[1]

## testia vaa
def prosessi_3(jono2):
    Col_list = []
    listan_pituus = jono2.get()
    for x in range(listan_pituus):
        osa = jono2.get()
        Col_list.append(osa)
    print(Col_list)
def prosessi_4(hk0, hk2):
    while True:
        print(hk0.value, hk2.value)
####

##### toisella corella lattia y ettiminen ja collision detect
def etitaangrid_multi(jono, hk0, hk1, hk2, sulje):
    Clock = pygame.time.Clock()
    lattia_grids = jono.get()
    scale = jono.get()
    hahmo_koords = [0.0, 0.0, 0.0]
    # + +
    while True:

        hahmo_koords[0] = -hk0.value
        hahmo_koords[2] = -hk2.value
        if sulje.value == 1:
            quit()

        y = 0
        #if hahmo_koords[1] < -11:
            #y = 1
        #print(y)


        if hahmo_koords[0] >= 0 and hahmo_koords[2] >= 0:

            if hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] >= 0 and hahmo_koords[
                2] <= 125 * scale:
                grid = lattia_grids[y][0]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[2] >= 0 and \
                            hahmo_koords[2] <= 125 * scale:
                grid = lattia_grids[y][1]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[2] >= 0 and \
                            hahmo_koords[2] <= 125 * scale:
                grid = lattia_grids[y][2]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[2] >= 0 and \
                            hahmo_koords[2] <= 125 * scale:
                grid = lattia_grids[y][3]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] >= 125 * scale and \
                            hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][4]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[
                2] >= 125 * scale and hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][5]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[
                2] >= 125 * scale and hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][6]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[
                2] >= 125 * scale and hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][7]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] >= 250 * scale and \
                            hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][8]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[
                2] >= 250 * scale and hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][9]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[
                2] >= 250 * scale and hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][10]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[
                2] >= 250 * scale and hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][11]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] >= 375 * scale and \
                            hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][12]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[
                2] >= 375 * scale and hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][13]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[
                2] >= 375 * scale and hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][14]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[
                2] >= 375 * scale and hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][15]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

        # + -
        elif hahmo_koords[0] >= 0 and hahmo_koords[2] <= 0:

            if hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] <= 0 and hahmo_koords[
                2] >= -125 * scale:
                grid = lattia_grids[y][16]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[2] <= 0 and \
                            hahmo_koords[2] >= -125 * scale:
                grid = lattia_grids[y][17]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[2] <= 0 and \
                            hahmo_koords[2] >= -125 * scale:
                grid = lattia_grids[y][18]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[2] <= 0 and \
                            hahmo_koords[2] >= -125 * scale:
                grid = lattia_grids[y][19]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] <= -125 * scale and \
                            hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][20]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[
                2] <= -125 * scale and hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][21]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[
                2] <= -125 * scale and hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][22]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[
                2] <= -125 * scale and hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][23]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] <= -250 * scale and \
                            hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][24]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[
                2] <= -250 * scale and hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][25]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[
                2] <= -250 * scale and hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][26]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[
                2] <= -250 * scale and hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][27]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] >= 0 and hahmo_koords[0] <= 125 * scale and hahmo_koords[2] <= -375 * scale and \
                            hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][28]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 125 * scale and hahmo_koords[0] <= 250 * scale and hahmo_koords[
                2] <= -375 * scale and hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][29]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 250 * scale and hahmo_koords[0] <= 375 * scale and hahmo_koords[
                2] <= -375 * scale and hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][30]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] >= 375 * scale and hahmo_koords[0] <= 500 * scale and hahmo_koords[
                2] <= -375 * scale and hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][31]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

        # - -
        elif hahmo_koords[0] <= 0 and hahmo_koords[2] <= 0:

            if hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] <= 0 and hahmo_koords[
                2] >= -125 * scale:
                grid = lattia_grids[y][32]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[2] <= 0 and \
                            hahmo_koords[2] >= -125 * scale:
                grid = lattia_grids[y][33]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[2] <= 0 and \
                            hahmo_koords[2] >= -125 * scale:
                grid = lattia_grids[y][34]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[2] <= 0 and \
                            hahmo_koords[2] >= -125 * scale:
                grid = lattia_grids[y][35]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] <= -125 * scale and \
                            hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][36]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[
                2] <= -125 * scale and hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][37]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[
                2] <= -125 * scale and hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][38]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[
                2] <= -125 * scale and hahmo_koords[2] >= -250 * scale:
                grid = lattia_grids[y][39]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] <= -250 * scale and \
                            hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][40]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[
                2] <= -250 * scale and hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][41]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[
                2] <= -250 * scale and hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][42]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[
                2] <= -250 * scale and hahmo_koords[2] >= -375 * scale:
                grid = lattia_grids[y][43]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] <= -375 * scale and \
                            hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][44]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[
                2] <= -375 * scale and hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][45]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[
                2] <= -375 * scale and hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][46]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[
                2] <= -375 * scale and hahmo_koords[2] >= -500 * scale:
                grid = lattia_grids[y][47]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

        # - +
        elif hahmo_koords[0] <= 0 and hahmo_koords[2] >= 0:

            if hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] >= 0 and hahmo_koords[
                2] <= 125 * scale:
                grid = lattia_grids[y][48]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[2] >= 0 and \
                            hahmo_koords[2] <= 125 * scale:
                grid = lattia_grids[y][49]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[2] >= 0 and \
                            hahmo_koords[2] <= 125 * scale:
                grid = lattia_grids[y][50]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[2] >= 0 and \
                            hahmo_koords[2] <= 125 * scale:
                grid = lattia_grids[y][51]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] >= 125 * scale and \
                            hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][52]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[
                2] >= 125 * scale and hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][53]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[
                2] >= 125 * scale and hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][54]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[
                2] >= 125 * scale and hahmo_koords[2] <= 250 * scale:
                grid = lattia_grids[y][55]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] >= 250 * scale and \
                            hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][56]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[
                2] >= 250 * scale and hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][57]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[
                2] >= 250 * scale and hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][58]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[
                2] >= 250 * scale and hahmo_koords[2] <= 375 * scale:
                grid = lattia_grids[y][59]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

            elif hahmo_koords[0] <= 0 and hahmo_koords[0] >= -125 * scale and hahmo_koords[2] >= 375 * scale and \
                            hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][60]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -125 * scale and hahmo_koords[0] >= -250 * scale and hahmo_koords[
                2] >= 375 * scale and hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][61]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -250 * scale and hahmo_koords[0] >= -375 * scale and hahmo_koords[
                2] >= 375 * scale and hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][62]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)
            elif hahmo_koords[0] <= -375 * scale and hahmo_koords[0] >= -500 * scale and hahmo_koords[
                2] >= 375 * scale and hahmo_koords[2] <= 500 * scale:
                grid = lattia_grids[y][63]
                lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1)

        Clock.tick(200)
def lahin_lattia_y_multi(grid, hahmo_koords, scale, hk1):
    hahmo_koords[0] = -hahmo_koords[0]
    hahmo_koords[2] = -hahmo_koords[2]

    skaala = scale
    edellinen_tarkastelu = 1000

    for osio in grid:
        X = float(osio[0] * skaala) + hahmo_koords[0]
        Z = float(osio[2] * skaala) + hahmo_koords[2]
        tarkastelu_summa = abs(X) + abs(Z)
        if tarkastelu_summa <= edellinen_tarkastelu:
            edellinen_tarkastelu = tarkastelu_summa
            lahin = -(osio[1] * skaala)
    hk1.value = lahin
def collision_check_multi(jono2, hk0, hk2, hahmoY, collision, Ctype, deny_jump,sulje,hcy):
    Clock = pygame.time.Clock()
    Col_list = []
    listan_pituus = jono2.get()
    for x in range(listan_pituus):
        osa = jono2.get()
        Col_list.append(osa)
    hahmo_koords = [0.0, 0.0, 0.0]

    hypyn_esto = 0
    coltype = 0 #0=ei 1=paalla 2=alla 3=sivu
    colcount = False

    while True:
        hahmo_koords[0] = hk0.value
        hahmo_koords[1] = hahmoY.value
        hahmo_koords[2] = hk2.value

        if sulje.value == 1:
            quit()

        for x in range(0, len(Col_list), 3):
            if math.sqrt(((-hahmo_koords[0] - Col_list[x][0]) ** 2) + \
                                 ((-hahmo_koords[2] - Col_list[x][2]) ** 2)) < Col_list[
                        x + 1]:
                if Col_list[x+2] == 0:
                    coltype = 3
                    colcount = True
                    break
                elif -hahmo_koords[1] - 1.5 > Col_list[x][1]:
                    hypyn_esto = 1
                    if hahmo_koords[1] > -Col_list[x][1] - Col_list[x + 1] - 2.0 and \
                                    hahmo_koords[1] < -Col_list[x][1] - Col_list[x + 1]:
                        hypyn_esto = 0
                        hcy.value = -Col_list[x][1] - Col_list[x + 1] - 2.0
                        coltype = 1

                    colcount = True
                    break

                elif -hahmo_koords[1] < Col_list[x][1]:
                    if -hahmo_koords[1] > Col_list[x][1] - Col_list[x + 1]:
                        hcy.value = -Col_list[x][1] + Col_list[x + 1]+0.1
                        coltype = 2
                        colcount = True
                        break
                    coltype = 0
                    colcount = False
                    break

                else:
                    coltype = 3
                    colcount = True
                    break
            else:
                hypyn_esto = 0
                coltype = 0
                colcount = False


        if colcount:
            collision.value = 1
        else:
            collision.value = 0


        Ctype.value = coltype
        deny_jump.value = hypyn_esto
        Clock.tick(75)

#shaderit
def normi_shader():
    vertex_shader = """
        #version 460
        layout ( location = 0 ) in vec3 position;
        layout ( location = 1 ) in vec2 tex;
        layout ( location = 2 ) in vec3 normal;
        layout ( location = 3 ) in vec3 tangent;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 transform;

        const vec4 plane = vec4(0,1,0,0);

        out vec2 newTexture;
        out vec3 FragPos;
        out mat3 TBN;

        void main()
        {

            vec4 v = vec4(position,1.0f);
            gl_Position = projection * view * model * transform * v;
            FragPos = vec3(model *  transform *  v);

            gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);

            ///NORMALMAPPINGSHIT
            vec3 T = normalize(model * transform * vec4(tangent, 0.0f)).xyz;
            vec3 N = normalize(model * transform * vec4(normal, 0.0f)).xyz;
            T = normalize(T - dot(T, N) * N);
            vec3 B = normalize(cross(N, T));
            TBN = mat3(T, B, N);

            newTexture = tex;
        }
        """

    fragment_shader = """
        #version 460
        in vec2 newTexture;
        in vec3 FragPos;
        in mat3 TBN;

        uniform mat4 lightpos;
        uniform mat4 viewPos;
        uniform sampler2D samplerTexture;
        uniform sampler2D normalMap;
        uniform float luhtu;
        uniform float specularStrenght;


        vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
        vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
        vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;
        uniform float ambientSTR;
        vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;

        vec4 Lightpos2 = vec4(49.0, 4.0, 9.0, 1.0);
        vec3 lightColor2 = vec3(1.0f,0.0f,0.0f);

        vec4 Lightpos3 = vec4(500.0, 500.0, 500.0, 1.0);
        uniform float sunlightSTR;
        vec3 lightColor3 = vec3(0.9f,0.55f,0.2f)*sunlightSTR;

        ///valofeidiparametreja
        float constant = 1.0;
        float linear = 0.09;
        float quadratic = 0.032;



        out vec4 outColor;
        void main()
        {


            ///NORMALS TO WORLDSPACE
            vec3 Normal = normalize(texture(normalMap, newTexture).xyz * 2.0 - 1.0);
            Normal = normalize(TBN * Normal);

            ///DIFFUSE
            vec3 norm = normalize(vec3(Normal));
            vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
            float diff2 = max(dot(norm, lightDir2), 0.0);
            vec3 diffuse2 = diff2 * lightColor2;

            vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
            float diff3 = max(dot(norm, lightDir3), 0.0);
            vec3 diffuse3 = diff3 * lightColor3;

            ////SPECULAR
            vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrenght * spec * lightColor;

            vec3 reflectDir2 = reflect(-lightDir2, norm);
            float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
            vec3 specular2 = specularStrenght * spec2 * lightColor2;

            vec3 reflectDir3 = reflect(-lightDir3, norm);
            float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), 32);
            vec3 specular3 = specularStrenght * spec3 * lightColor3;

            //ATTENUATION
            float distance    = length(Lightpos.xyz - FragPos);
            float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

            float distance2    = length(Lightpos2.xyz - FragPos);
            float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

            float distance3    = length(Lightpos3.xyz - FragPos);
            float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

            ///VALOSUMMA
            vec3 result = (diffuse  + specular) * attenuation;
            vec3 result2 = (diffuse2 + specular2) * attenuation2;
            vec3 result3 = (diffuse3 + specular3);

            vec3 totalResult = result + result2 + result3 + ambient;

            ///TEXTUURI
            vec4 texel = texture2D(samplerTexture, newTexture);



            ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
            outColor = vec4(texel) * vec4(totalResult, 1.0f);    //*vec4(newColor, 1.0f);
        }
        """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                   OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                   GL_FRAGMENT_SHADER))
    return shader
def terrain_nomulti_shader():
    vertex_shader = """
        #version 460
        layout ( location = 0 ) in vec3 position;
        layout ( location = 1 ) in vec2 tex;
        layout ( location = 2 ) in vec3 normal;
        layout ( location = 3 ) in vec3 tangent;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 transform;

        const vec4 plane = vec4(0,1,0,0);

        out vec2 newTexture;
        out vec3 FragPos;
        out mat3 TBN;

        void main()
        {

            vec4 v = vec4(position,1.0f);
            gl_Position = projection * view * model * transform * v;
            FragPos = vec3(model *  transform *  v);

            gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);

            ///NORMALMAPPINGSHIT
            vec3 T = normalize(model * transform * vec4(tangent, 0.0f)).xyz;
            vec3 N = normalize(model * transform * vec4(normal, 0.0f)).xyz;
            T = normalize(T - dot(T, N) * N);
            vec3 B = normalize(cross(N, T));
            TBN = mat3(T, B, N);

            newTexture = tex;
        }
        """

    fragment_shader = """
        #version 460
        in vec2 newTexture;
        in vec3 FragPos;
        in mat3 TBN;

        uniform mat4 lightpos;
        uniform mat4 viewPos;
        uniform sampler2D samplerTexture;
        uniform sampler2D normalMap;
        uniform float luhtu;
        uniform float specularStrenght;


        vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
        vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
        vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;

        uniform float ambientSTR;
        vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;

        vec4 Lightpos2 = vec4(49.0, 4.0, 9.0, 1.0);
        vec3 lightColor2 = vec3(1.0f,0.0f,0.0f);

        vec4 Lightpos3 = vec4(0.0, 500.0, 0.0, 1.0);
        uniform float sunlightSTR;
        vec3 lightColor3 = vec3(0.9f,0.55f,0.2f)*sunlightSTR;

        ///valofeidiparametreja
        float constant = 1.0;
        float linear = 0.09;
        float quadratic = 0.032;



        out vec4 outColor;
        void main()
        {
            vec2 skaledtex = newTexture * 120.0;

            ///NORMALS TO WORLDSPACE
            vec3 Normal = normalize(texture(normalMap, skaledtex).xyz * 2.0 - 1.0);
            Normal = normalize(TBN * Normal);

            ///DIFFUSE
            vec3 norm = normalize(vec3(Normal));
            vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
            float diff2 = max(dot(norm, lightDir2), 0.0);
            vec3 diffuse2 = diff2 * lightColor2;

            vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
            float diff3 = max(dot(norm, lightDir3), 0.0);
            vec3 diffuse3 = diff3 * lightColor3;

            ////SPECULAR
            vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrenght * spec * lightColor;

            vec3 reflectDir2 = reflect(-lightDir2, norm);
            float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
            vec3 specular2 = specularStrenght * spec2 * lightColor2;

            vec3 reflectDir3 = reflect(-lightDir3, norm);
            float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), 32);
            vec3 specular3 = specularStrenght * spec3 * lightColor3;

            //ATTENUATION
            float distance    = length(Lightpos.xyz - FragPos);
            float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

            float distance2    = length(Lightpos2.xyz - FragPos);
            float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

            float distance3    = length(Lightpos3.xyz - FragPos);
            float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

            ///VALOSUMMA
            vec3 result = (diffuse  + specular) * attenuation;
            vec3 result2 = (diffuse2 + specular2) * attenuation2;
            vec3 result3 = (diffuse3 + specular3);

            vec3 totalResult = result + result2 + result3 + ambient;

            ///TEXTUURI
            vec4 texel = texture2D(samplerTexture, skaledtex);



            ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
            outColor = vec4(texel) * vec4(totalResult, 1.0f);    //*vec4(newColor, 1.0f);
        }
        """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                   OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                   GL_FRAGMENT_SHADER))
    return shader
def terrain_shader():
    vertex_shader = """
        #version 460
        layout ( location = 0 ) in vec3 position;
        layout ( location = 1 ) in vec2 tex;
        layout ( location = 2 ) in vec3 normal;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 transform;

        const vec4 plane = vec4(0,1,0,0);

        out vec2 newTexture;
        out vec3 FragPos;
        out vec3 Norm;

        void main()
        {

            vec4 v = vec4(position,1.0f);
            gl_Position = projection * view * model * transform * v;
            FragPos = vec3(model *  transform *  v);

            gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);

            Norm = normal;
            newTexture = tex;
        }
        """

    fragment_shader = """
        #version 460
        in vec2 newTexture;
        in vec3 FragPos;
        in vec3 Norm;

        uniform mat4 lightpos;
        uniform mat4 viewPos;

        uniform sampler2D blendMap;
        uniform sampler2D BGTex;
        uniform sampler2D rTex;
        uniform sampler2D gTex;
        uniform sampler2D bTex;

        uniform sampler2D BGnormalMap;
        uniform sampler2D RnormalMap;
        uniform sampler2D GnormalMap;
        uniform sampler2D BnormalMap;


        uniform float luhtu;
        uniform float specularStrenght;

        vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
        vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
        vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;
        uniform float ambientSTR;
        vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;

        vec4 Lightpos2 = vec4(49.0, 4.0, 9.0, 1.0);
        vec3 lightColor2 = vec3(1.0f,0.0f,0.0f);

        vec4 Lightpos3 = vec4(500.0, 500.0, 500.0, 1.0);
        uniform float sunlightSTR;
        vec3 lightColor3 = vec3(0.5f,0.5f,0.5f)*sunlightSTR;

        ///valofeidiparametreja
        float constant = 1.0;
        float linear = 0.09;
        float quadratic = 0.032;



        out vec4 outColor;
        void main()
        {


            vec4 blendMapColor = texture(blendMap, newTexture);
            float BGtexAmount = 1 - (blendMapColor.r + blendMapColor.g + blendMapColor.b);
            vec2 texdensity = newTexture * 120.0;
            vec4 BGtexColor = texture(BGTex, texdensity)*BGtexAmount;
            vec4 rtexColor = texture(rTex, texdensity)*blendMapColor.r;
            vec4 gtexColor = texture(gTex, texdensity)*blendMapColor.g;
            vec4 btexColor = texture(bTex, texdensity)*blendMapColor.b;
            vec4 totalColor = BGtexColor + rtexColor + gtexColor + btexColor;


            vec4 BGnormal = texture(BGnormalMap, texdensity) * BGtexAmount;
            vec4 rnormal = texture(RnormalMap, texdensity) * blendMapColor.r;
            vec4 gnormal = texture(GnormalMap, texdensity) * blendMapColor.g;
            vec4 bnormal = texture(BnormalMap, texdensity) * blendMapColor.b;
            vec4 totalnormal = BGnormal + rnormal + gnormal + bnormal;




            // compute tangent T and bitangent B
            vec3 Q1 = dFdx(FragPos);
            vec3 Q2 = dFdy(FragPos);
            vec2 st1 = dFdx(texdensity);
            vec2 st2 = dFdy(texdensity);

            vec3 T = normalize(Q1*st2.t - Q2*st1.t);
            vec3 B = normalize(-Q1*st2.s + Q2*st1.s);
            vec3 N = normalize(Norm);

            // the transpose of texture-to-eye space matrix
            mat3 TBN = mat3(T, B, N);


            vec3 Normal = normalize(totalnormal.xyz * 2.0 - 1.0);
            Normal = (TBN * Normal);




            ///DIFFUSE
            vec3 norm = normalize(vec3(Normal));
            vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
            float diff2 = max(dot(norm, lightDir2), 0.0);
            vec3 diffuse2 = diff2 * lightColor2;

            vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
            float diff3 = max(dot(norm, lightDir3), 0.0);
            vec3 diffuse3 = diff3 * lightColor3;

            ////SPECULAR
            vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrenght * spec * lightColor;

            vec3 reflectDir2 = reflect(-lightDir2, norm);
            float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
            vec3 specular2 = specularStrenght * spec2 * lightColor2;

            vec3 reflectDir3 = reflect(-lightDir3, norm);
            float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), 32);
            vec3 specular3 = specularStrenght * spec3 * lightColor3;

            //ATTENUATION
            float distance    = length(Lightpos.xyz - FragPos);
            float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

            float distance2    = length(Lightpos2.xyz - FragPos);
            float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

            float distance3    = length(Lightpos3.xyz - FragPos);
            float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

            ///VALOSUMMA
            vec3 result = (diffuse  + specular) * attenuation;
            vec3 result2 = (diffuse2 + specular2) * attenuation2;
            vec3 result3 = (diffuse3 + specular3);

            vec3 totalResult = result + result2 + result3 + ambient;



            ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
            outColor = vec4(totalColor) * vec4(totalResult, 1.0f);
        }
        """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                   OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                   GL_FRAGMENT_SHADER))
    return shader
def nolight_shader():
    vertex_shader = """
                #version 460
                layout ( location = 0 ) in vec3 position;
                layout ( location = 1 ) in vec2 tex;
                layout ( location = 2 ) in vec3 normal;

                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform mat4 transform;

                out vec2 newTexture;


                void main()
                {
                    vec4 v = vec4(position,1.0f);
                    vec3 FragPos = vec3(model *  transform *  v);

                    gl_Position = projection * view * model * transform * v;
                    newTexture = tex;
                }
                """

    fragment_shader = """
                #version 460
                in vec2 newTexture;

                uniform sampler2D samplerTexture;
                uniform float sunStr;


                out vec4 outColor;
                void main()
                {
                    ///TEXTUURI
                    vec4 texel = texture2D(samplerTexture, newTexture);
                    float pimeys = 0.3/sunStr;
                    vec4 C = vec4(0.15,0.07,0.0,1.0);

                    float avg = (texel.r+texel.g+texel.b)/3.0;
                    vec4 GS = vec4(avg,avg,avg,1.0);
                    vec4 color = GS * C;

                    if (pimeys > 1.0){
                        outColor = mix(vec4(texel),vec4(color),1.0)/pimeys;
                    }else{
                        outColor = mix(vec4(texel),vec4(color),pimeys);
                    }


                }
                """

    shader2 = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return shader2
def instanssi_shaderi():
    vertex_shader = """
                #version 460
                in layout ( location = 0 ) vec3 position;
                in layout ( location = 1 ) vec2 tex;
                in layout ( location = 2 ) vec3 normal;
                in layout ( location = 3 ) vec3 tangent;
                in layout ( location = 4 ) mat4 instanceMatrix;
                in layout ( location = 8 ) mat4 instanceTrans;


                uniform mat4 projection;
                uniform mat4 view;
                uniform float time;

                const vec4 plane = vec4(0,1,0,0);

                out vec2 newTexture;
                out vec3 FragPos;
                out mat3 TBN;

                void main()
                {
                    vec4 v = vec4(position,1.0f);
                    gl_Position = projection * view * instanceMatrix * instanceTrans * v;
                    FragPos = vec3(instanceMatrix *instanceTrans *  v);

                    gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);

                    ///NORMALMAPPINGSHIT
                    vec3 T = normalize(instanceMatrix *instanceTrans * vec4(tangent, 0.0f)).xyz;
                    vec3 N = normalize(instanceMatrix *instanceTrans * vec4(normal, 0.0f)).xyz;
                    T = normalize(T - dot(T, N) * N);
                    vec3 B = normalize(cross(N, T));
                    TBN = mat3(T, B, N);

                    newTexture = tex;
                }
                """

    fragment_shader = """
                #version 460
                in vec2 newTexture;
                in vec3 FragPos;
                in mat3 TBN;

                uniform mat4 lightpos;
                uniform mat4 viewPos;
                uniform sampler2D samplerTexture;
                uniform sampler2D normalMap;
                uniform float luhtu;
                uniform float lapi;
                uniform float specularStrenght;


                vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
                vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
                vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;
                uniform float ambientSTR;
                vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;

                vec4 Lightpos2 = vec4(49.0, 4.0, 9.0, 1.0);
                vec3 lightColor2 = vec3(1.0f,0.0f,0.0f);

                vec4 Lightpos3 = vec4(500.0, 500.0, 500.0, 1.0);
                uniform float sunlightSTR;
                vec3 lightColor3 = vec3(0.5f,0.5f,0.5f)*sunlightSTR;


                float constant = 1.0;
                float linear = 0.09;
                float quadratic = 0.032;


                out vec4 outColor;
                void main()
                {


                    ///NORMALS TO WORLDSPACE
                    vec3 Normal = normalize(texture(normalMap, newTexture).xyz * 2.0 - 1.0);
                    Normal = normalize(TBN * Normal);

                    ///DIFFUSE
                    vec3 norm = normalize(vec3(Normal));
                    vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
                    float diff = max(dot(norm, lightDir), 0.0);
                    vec3 diffuse = diff * lightColor;

                    vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
                    float diff2 = max(dot(norm, lightDir2), 0.0);
                    vec3 diffuse2 = diff2 * lightColor2;

                    vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
                    float diff3 = max(dot(norm, lightDir3), 0.0);
                    vec3 diffuse3 = diff3 * lightColor3;


                    vec3 norm9 = normalize(vec3(-Normal));

                    float diff9 = max(dot(norm9, lightDir), 0.0);
                    vec3 diffuse9 = diff9 * lightColor;


                    float diff29 = max(dot(norm9, lightDir2), 0.0);
                    vec3 diffuse29 = diff29 * lightColor2;


                    float diff39 = max(dot(norm9, lightDir3), 0.0);
                    vec3 diffuse39 = diff39 * lightColor3;





                    ////SPECULAR
                    vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
                    vec3 reflectDir = reflect(-lightDir, norm);
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
                    vec3 specular = specularStrenght * spec * lightColor;

                    vec3 reflectDir2 = reflect(-lightDir2, norm);
                    float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
                    vec3 specular2 = specularStrenght * spec2 * lightColor2;

                    vec3 reflectDir3 = reflect(-lightDir3, norm);
                    float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), 32);
                    vec3 specular3 = specularStrenght * spec3 * lightColor3;

                    //ATTENUATION
                    float distance    = length(Lightpos.xyz - FragPos);
                    float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

                    float distance2    = length(Lightpos2.xyz - FragPos);
                    float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

                    float distance3    = length(Lightpos3.xyz - FragPos);
                    float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

                    ///VALOSUMMA
                    vec3 result = (diffuse  + specular + diffuse9) * attenuation;
                    vec3 result2 = (diffuse2 + specular2 + diffuse29) * attenuation2;
                    vec3 result3 = (diffuse3 + specular3 + diffuse39);

                    vec3 totalResult = result + result2 + result3 + ambient;

                    ///TEXTUURI
                    vec4 texel = texture2D(samplerTexture, newTexture);
                    if(texel.a < 0.5)
                    {
                        discard;
                    }

                    ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
                    outColor = texel * vec4(totalResult, 1.0f);

                }
                """

    shader_instance = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return shader_instance
def instanssi_shaderi_ref():
    vertex_shader = """
                #version 460
                in layout ( location = 0 ) vec3 position;
                in layout ( location = 1 ) vec2 tex;
                in layout ( location = 2 ) vec3 normal;
                in layout ( location = 4 ) mat4 instanceMatrix;
                in layout ( location = 8 ) mat4 instanceTrans;


                uniform mat4 projection;
                uniform mat4 view;

                const vec4 plane = vec4(0,1,0,0);

                out vec2 newTexture;
                out vec3 FragPos;
                out vec3 Norm;

                void main()
                {
                    vec4 v = vec4(position,1.0f);
                    gl_Position = projection * view * instanceMatrix * instanceTrans * v;
                    FragPos = vec3(instanceMatrix *instanceTrans *  v);

                    gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);

                    Norm = normal;
                    newTexture = tex;
                }
                """

    fragment_shader = """
                #version 460
                in vec2 newTexture;
                in vec3 FragPos;
                in vec3 Norm;

                uniform mat4 lightpos;
                uniform mat4 viewPos;
                uniform sampler2D samplerTexture;
                uniform float luhtu;


                vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
                vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
                vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;
                uniform float ambientSTR;
                vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;

                vec4 Lightpos2 = vec4(49.0, 4.0, 9.0, 1.0);
                vec3 lightColor2 = vec3(1.0f,0.0f,0.0f);

                vec4 Lightpos3 = vec4(500.0, 500.0, 500.0, 1.0);
                uniform float sunlightSTR;
                vec3 lightColor3 = vec3(0.5f,0.5f,0.5f)*sunlightSTR;


                float constant = 1.0;
                float linear = 0.09;
                float quadratic = 0.032;


                out vec4 outColor;
                void main()
                {



                    ///DIFFUSE
                    vec3 norm = normalize(vec3(Norm));
                    vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
                    float diff = max(dot(norm, lightDir), 0.0);
                    vec3 diffuse = diff * lightColor;

                    vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
                    float diff2 = max(dot(norm, lightDir2), 0.0);
                    vec3 diffuse2 = diff2 * lightColor2;

                    vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
                    float diff3 = max(dot(norm, lightDir3), 0.0);
                    vec3 diffuse3 = diff3 * lightColor3;


                    vec3 norm9 = normalize(vec3(-Norm));

                    float diff9 = max(dot(norm9, lightDir), 0.0);
                    vec3 diffuse9 = diff9 * lightColor;


                    float diff29 = max(dot(norm9, lightDir2), 0.0);
                    vec3 diffuse29 = diff29 * lightColor2;


                    float diff39 = max(dot(norm9, lightDir3), 0.0);
                    vec3 diffuse39 = diff39 * lightColor3;


                    //ATTENUATION
                    float distance    = length(Lightpos.xyz - FragPos);
                    float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

                    float distance2    = length(Lightpos2.xyz - FragPos);
                    float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

                    float distance3    = length(Lightpos3.xyz - FragPos);
                    float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

                    ///VALOSUMMA
                    vec3 result = (diffuse + diffuse9) * attenuation;
                    vec3 result2 = (diffuse2 + diffuse29) * attenuation2;
                    vec3 result3 = (diffuse3 + diffuse39);

                    vec3 totalResult = result + result2 + result3 + ambient;

                    ///TEXTUURI
                    vec4 texel = texture2D(samplerTexture, newTexture);
                    if(texel.a < 0.5)
                    {
                        discard;
                    }

                    ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
                    outColor = texel * vec4(totalResult, 1.0f);

                }
                """

    shader_instance = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return shader_instance
def depth_shader():
    vertex_shader = """
                #version 440
                layout ( location = 0 ) in vec3 position;

                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform mat4 transform;

                void main()
                {
                    vec4 v = vec4(position,1.0f);
                    gl_Position = projection * view * model * transform * v;
                }
                """

    fragment_shader = """
                #version 440

                void main()
                {
                    gl_FragDepth = gl_FragCoord.z;
                }
                """

    depth_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return depth_shader
def shadow_shader():
    vertex_shader = """
        #version 440
        layout ( location = 0 ) in vec3 position;
        layout ( location = 1 ) in vec2 tex;
        layout ( location = 2 ) in vec3 normal;
        layout ( location = 3 ) in vec3 color;
        layout ( location = 4 ) in vec3 tangent;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 transform;


        out vec3 newColor;
        out vec2 newTexture;
        out vec3 FragPos;
        out mat3 TBN;

        void main()
        {

            vec4 v = vec4(position,1.0f);
            gl_Position = projection * view * model * transform * v;
            FragPos = vec3(model *  transform *  v);

            ///NORMALMAPPINGSHIT
            vec3 T = normalize(model * transform * vec4(tangent, 0.0f)).xyz;
            vec3 N = normalize(model * transform * vec4(normal, 0.0f)).xyz;
            T = normalize(T - dot(T, N) * N);
            vec3 B = normalize(cross(N, T));
            TBN = mat3(T, B, N);

            newColor = color;
            newTexture = tex;
        }
        """

    fragment_shader = """
        #version 440
        in vec3 newColor;
        in vec2 newTexture;
        in vec3 FragPos;
        in mat3 TBN;

        uniform mat4 lightpos;
        uniform mat4 viewPos;
        uniform sampler2D samplerTexture;
        uniform sampler2D normalMap;
        uniform sampler2D shadowMap;
        uniform float luhtu;
        uniform float specularStrenght;

        /////JONKUN RUMAN HUORAN FOGSHAISSE
        float getFogFactor(float d)
        {
            const float FogMax = 25.0;
            const float FogMin = 5.0;

            if (d>=FogMax) return 1;
            if (d<=FogMin) return 0;

            return 1 - (FogMax - d) / (FogMax - FogMin);
        }

        vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
        vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
        vec4 FogColor = vec4(0.0,0.0,0.0,0.0);
        vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;
        vec3 ambient = vec3(0.15f,0.15f,0.2f);

        vec4 fragPosLightSpace = vec4(FragPos,1.0);

        float ShadowCalculation(vec4 fragPosLightSpace)
        {
            vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
            projCoords = projCoords * 0.5 + 0.5;
            float closestDepth = texture(shadowMap, projCoords.xy).r;
            float currentDepth = projCoords.z;
            float shadow = currentDepth > closestDepth  ? 1.0 : 0.0;
            return shadow;
        }


        out vec4 outColor;
        void main()
        {


            ///NORMALS TO WORLDSPACE
            vec3 Normal = normalize(texture(normalMap, newTexture).xyz * 2.0 - 1.0);
            Normal = normalize(TBN * Normal);

            ///DIFFUSE
            vec3 norm = normalize(vec3(Normal));
            vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            ////SPECULAR
            vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrenght * spec * lightColor;

            ///VARJO
            float shadow = ShadowCalculation(fragPosLightSpace);

            ///VALOSUMMA
            vec3 result = ambient +(1.0-shadow) * (diffuse + specular);

            ///TEXTUURI
            vec4 texel = texture2D(samplerTexture, newTexture);

            ////FOCK
            vec4 V = vec4(FragPos,0.0f);
            float d = distance(Lightpos, V);
            float alpha = getFogFactor(d);

            ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS

            outColor = vec4(texel) * vec4(result, 1.0f);    //*vec4(newColor, 1.0f);
            outColor = mix(outColor, FogColor, alpha);
        }
        """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                   OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                   GL_FRAGMENT_SHADER))
    return shader
def reflect_shader():
    vertex_shader = """
                #version 460
                layout ( location = 0 ) in vec3 position;
                layout ( location = 1 ) in vec2 tex;
                layout ( location = 2 ) in vec3 normal;

                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform mat4 transform;

                out vec2 newTexture;
                out vec4 clipspace;

                void main()
                {

                    vec4 v = vec4(position,1.0f);
                    clipspace = projection * view * model * transform * v;
                    gl_Position = clipspace;
                    newTexture = tex;
                }
                """

    fragment_shader = """
                #version 460
                in vec2 newTexture;

                in vec4 clipspace;

                uniform sampler2D samplerTexture;
                uniform sampler2D dudvMap;
                uniform float move;

                const float disto = 0.01;


                out vec4 outColor;
                void main()
                {


                    vec2 ndc = (clipspace.xy/clipspace.w)/2.0f + 0.5f;
                    vec2 reflectTexCoords = vec2(ndc.x, -ndc.y);

                    vec2 distortion = (texture2D(dudvMap, vec2(newTexture.x + move,newTexture.y)).rg*2.0-1.0)*disto;
                    vec2 distortion2 = (texture2D(dudvMap, vec2(-newTexture.x + move,newTexture.y+move)).rg*2.0-1.0)*disto;
                    vec2 distoKOK = distortion + distortion2;

                    reflectTexCoords += distoKOK;
                    reflectTexCoords.x = clamp(reflectTexCoords.x, 0.001,0.999);
                    reflectTexCoords.y = clamp(reflectTexCoords.y, -0.999,-0.001);

                    vec4 texel = texture2D(samplerTexture, reflectTexCoords);
                    outColor = vec4(texel);//* vec4(newColor, 1.0f);
                    //outColor = mix(outColor, vec4(0.0, 0.4, 0.7, 1.0), 0.2);
                }
                """

    shader2 = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return shader2
def final_shader():
    vertex_shader = """
                        #version 460
                        layout ( location = 0 ) in vec3 position;
                        layout ( location = 1 ) in vec2 tex;

                        out vec2 newtex;

                        void main()
                        {
                            gl_Position = vec4(position,1.0f);
                            newtex = tex;
                        }
                        """

    fragment_shader = """
                        #version 460
                        in vec2 newtex;

                        uniform sampler2D samplerTexture;
                        uniform sampler2D toinen;
                        uniform sampler2D prev1;
                        uniform int postprocess;
                        uniform int motionblur;
                        uniform int glow;

                        out vec4 outColor;
                        void main()
                        {
                            if (glow > 0.5){
                                if (postprocess > 0.5){
                                    vec4 texel = vec4(vec3(1.0 - texture2D(samplerTexture, newtex)), 1.0);
                                    vec4 texel2 = texture2D(toinen, newtex);
                                    vec4 texel3 = texture2D(samplerTexture, newtex);
                                    float avg = (texel3.r + texel3.g + texel3.b)/3.0;
                                    vec4 teX = vec4(avg,avg,avg,1.0);
                                    outColor = vec4(texel)*vec4(teX)*1.5+vec4(texel2);
                                    if (outColor.x > 0.1){
                                        outColor = vec4(outColor.x * 3, outColor.y, outColor.z, outColor.w);
                                    }else{
                                        outColor = vec4(outColor.x, outColor.y*3, outColor.z, outColor.w);
                                    }
                                    if (outColor.z < 0.001){
                                        outColor = vec4(outColor.x, outColor.y, outColor.z + 0.1, outColor.w);
                                    }
                                    if (motionblur > 0.5){
                                        vec4 texel4 = texture2D(prev1, newtex);
                                        outColor = mix(texel4, outColor, 0.3);
                                    }


                                }else{
                                    vec4 texel = texture2D(samplerTexture, newtex);
                                    vec4 texel2 = texture2D(toinen, newtex);
                                    outColor = vec4(texel)+vec4(texel2);
                                    if (motionblur > 0.5){
                                        vec4 texel4 = texture2D(prev1, newtex);
                                        outColor = mix(texel4, outColor, 0.3);
                                    }
                                }
                            }else{
                                if (postprocess > 0.5){
                                    vec4 texel = vec4(vec3(1.0 - texture2D(samplerTexture, newtex)), 1.0);
                                    //vec4 texel2 = texture2D(toinen, newtex);
                                    vec4 texel3 = texture2D(samplerTexture, newtex);
                                    float avg = (texel3.r + texel3.g + texel3.b)/3.0;
                                    vec4 teX = vec4(avg,avg,avg,1.0);
                                    outColor = vec4(texel)*vec4(teX)*1.5;
                                    if (outColor.x > 0.1){
                                        outColor = vec4(outColor.x * 3, outColor.y, outColor.z, outColor.w);
                                    }else{
                                        outColor = vec4(outColor.x, outColor.y*3, outColor.z, outColor.w);
                                    }
                                    if (outColor.z < 0.001){
                                        outColor = vec4(outColor.x, outColor.y, outColor.z + 0.1, outColor.w);
                                    }
                                    if (motionblur > 0.5){
                                        vec4 texel4 = texture2D(prev1, newtex);
                                        outColor = mix(texel4, outColor, 0.3);
                                    }


                                }else{
                                    vec4 texel = texture2D(samplerTexture, newtex);
                                    //vec4 texel2 = texture2D(toinen, newtex);
                                    outColor = vec4(texel);
                                    if (motionblur > 0.5){
                                        vec4 texel4 = texture2D(prev1, newtex);
                                        outColor = mix(texel4, outColor, 0.3);
                                    }
                                }
                            }
                        }
                        """

    final_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))

    return final_shader
def hor_blur():
    vertex_shader = """
                        #version 460
                        layout ( location = 0 ) in vec3 position;
                        //layout ( location = 1 ) in vec2 tex;


                        out vec2 BTC[11];
                        //out vec2 newtex;

                        void main()
                        {
                            gl_Position = vec4(position,1.0f);
                            vec2 Ctexcoords = position.xy * 0.5 +0.5;
                            float pixelSize = 1.0/320.0;

                            for(int i=-5;i<=5;i++)
                            {
                                BTC[i+5] = Ctexcoords + vec2(pixelSize * i, 0.0);
                            }

                            //newtex = tex;
                        }
                        """

    fragment_shader = """
                        #version 460
                        //in vec2 newtex;
                        in vec2 BTC[11];

                        uniform sampler2D samplerTexture;

                        out vec4 outColor;
                        void main()
                        {
                            //vec4 texel = texture2D(samplerTexture, newtex);
                            //outColor = vec4(vec3(1.0 - texture(samplerTexture, newtex)), 1.0);
                            outColor = vec4(0.0);
                            outColor += texture(samplerTexture, BTC[0]) * 0.0093;
                            outColor += texture(samplerTexture, BTC[1]) * 0.028;
                            outColor += texture(samplerTexture, BTC[2]) * 0.065984;
                            outColor += texture(samplerTexture, BTC[3]) * 0.1217;
                            outColor += texture(samplerTexture, BTC[4]) * 0.1757;
                            outColor += texture(samplerTexture, BTC[5]) * 0.19859;
                            outColor += texture(samplerTexture, BTC[6]) * 0.1757;
                            outColor += texture(samplerTexture, BTC[7]) * 0.1217;
                            outColor += texture(samplerTexture, BTC[8]) * 0.065984;
                            outColor += texture(samplerTexture, BTC[9]) * 0.028;
                            outColor += texture(samplerTexture, BTC[10]) * 0.0093;

                        }
                        """

    final_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))

    return final_shader
def ver_blur():
    vertex_shader = """
                        #version 460
                        layout ( location = 0 ) in vec3 position;
                        //layout ( location = 1 ) in vec2 tex;


                        out vec2 BTC[11];
                        //out vec2 newtex;

                        void main()
                        {
                            gl_Position = vec4(position,1.0f);
                            vec2 Ctexcoords = position.xy * 0.5 +0.5;
                            float pixelSize = 1.0/200.0;

                            for(int i=-5;i<=5;i++)
                            {
                                BTC[i+5] = Ctexcoords + vec2(0.0, pixelSize * i);
                            }

                            //newtex = tex;
                        }
                        """

    fragment_shader = """
                        #version 460
                        //in vec2 newtex;
                        in vec2 BTC[11];

                        uniform sampler2D samplerTexture;

                        out vec4 outColor;
                        void main()
                        {
                            //vec4 texel = texture2D(samplerTexture, newtex);
                            //outColor = vec4(vec3(1.0 - texture(samplerTexture, newtex)), 1.0);
                            outColor = vec4(0.0);
                            outColor += texture(samplerTexture, BTC[0]) * 0.0093;
                            outColor += texture(samplerTexture, BTC[1]) * 0.028;
                            outColor += texture(samplerTexture, BTC[2]) * 0.065984;
                            outColor += texture(samplerTexture, BTC[3]) * 0.1217;
                            outColor += texture(samplerTexture, BTC[4]) * 0.1757;
                            outColor += texture(samplerTexture, BTC[5]) * 0.19859;
                            outColor += texture(samplerTexture, BTC[6]) * 0.1757;
                            outColor += texture(samplerTexture, BTC[7]) * 0.1217;
                            outColor += texture(samplerTexture, BTC[8]) * 0.065984;
                            outColor += texture(samplerTexture, BTC[9]) * 0.028;
                            outColor += texture(samplerTexture, BTC[10]) * 0.0093;

                        }
                        """

    final_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))

    return final_shader
def brightspot_shader():
    vertex_shader = """
                        #version 460
                        layout ( location = 0 ) in vec3 position;
                        layout ( location = 1 ) in vec2 tex;

                        out vec2 newtex;

                        void main()
                        {
                            gl_Position = vec4(position,1.0f);
                            newtex = tex;
                        }
                        """

    fragment_shader = """
                        #version 460
                        in vec2 newtex;

                        uniform sampler2D samplerTexture;

                        out vec4 outColor;
                        void main()
                        {
                            vec4 texel = texture2D(samplerTexture, newtex);
                            float brightness = (texel.r) + (texel.g) + (texel.b);
                            if (brightness > 2.5){
                                outColor = vec4(texel);
                            }else{
                                outColor = vec4(0.0f);
                            }
                        }
                        """

    final_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))

    return final_shader
def motionblur_shader():
    vertex_shader = """
                            #version 460
                            layout ( location = 0 ) in vec3 position;
                            layout ( location = 1 ) in vec2 tex;

                            out vec2 newtex;

                            void main()
                            {
                                gl_Position = vec4(position,1.0f);
                                newtex = tex;
                            }
                            """

    fragment_shader = """
                            #version 460
                            in vec2 newtex;

                            uniform sampler2D samplerTexture;
                            uniform sampler2D toinen;
                            uniform sampler2D prev1;
                            uniform int postprocess;

                            out vec4 outColor;
                            void main()
                            {

                                if (postprocess > 0.5){
                                    vec4 texel = vec4(vec3(1.0 - texture2D(samplerTexture, newtex)), 1.0);
                                    vec4 texel2 = texture2D(toinen, newtex);
                                    vec4 texel3 = texture2D(samplerTexture, newtex);
                                    float avg = (texel3.r + texel3.g + texel3.b)/3.0;
                                    vec4 teX = vec4(avg,avg,avg,1.0);
                                    outColor = vec4(texel)*vec4(teX)*1.5+vec4(texel2);
                                    if (outColor.x > 0.1){
                                        outColor = vec4(outColor.x * 3, outColor.y, outColor.z, outColor.w);
                                    }else{
                                        outColor = vec4(outColor.x, outColor.y*3, outColor.z, outColor.w);
                                    }
                                    if (outColor.z < 0.001){
                                        outColor = vec4(outColor.x, outColor.y, outColor.z + 0.1, outColor.w);
                                    }
                                }else{
                                    vec4 texel = texture2D(samplerTexture, newtex);
                                    vec4 texel2 = texture2D(toinen, newtex);
                                    outColor = vec4(texel)+vec4(texel2);
                                }
                            }
                            """

    final_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))

    return final_shader
def instanssi_shaderi_wind():
    vertex_shader = """
                #version 460
                in layout ( location = 0 ) vec3 position;
                in layout ( location = 1 ) vec2 tex;
                in layout ( location = 2 ) vec3 normal;
                in layout ( location = 4 ) mat4 instanceMatrix;
                in layout ( location = 8 ) mat4 instanceTrans;
                in layout ( location = 12 ) float random;


                uniform mat4 projection;
                uniform mat4 view;
                uniform float time;
                uniform float Voima;


                const vec4 plane = vec4(0,1,0,0);


                out vec2 newTexture;
                out vec3 FragPos;
                out vec3 Norm;

                void main()
                {
                    vec3 offset = vec3( Voima * (cos( time/1.5+random *2 ) * (tex.y)) * sin( position.x),
                           0.25 * sin( 1.5*position.x + 2.0*position.z )* (tex.y),
                          Voima * (cos( time/2.0 +random*2 ) * (tex.y)) * sin( position.z)
                         );

                    vec4 v = vec4( position.x + offset.x, position.y + offset.y, position.z + offset.z + offset.x/5.0 + offset.y/2.0, 1.0);// vec4(position,1.0f);
                    gl_Position = projection * view * instanceMatrix * instanceTrans * v;
                    FragPos = vec3(instanceMatrix *instanceTrans *  v);

                    gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);


                    Norm = normal;
                    newTexture = tex;
                }
                """

    fragment_shader = """
                #version 460
                in vec2 newTexture;
                in vec3 FragPos;
                in vec3 Norm;

                uniform mat4 lightpos;
                uniform mat4 viewPos;
                uniform sampler2D samplerTexture;
                uniform sampler2D normalMap;
                uniform float luhtu;
                uniform float specularStrenght;


                vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
                vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
                vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;
                uniform float ambientSTR;
                vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;

                vec4 Lightpos2 = vec4(49.0, 4.0, 9.0, 1.0);
                vec3 lightColor2 = vec3(1.0f,0.0f,0.0f);

                vec4 Lightpos3 = vec4(500.0, 500.0, 500.0, 1.0);
                uniform float sunlightSTR;
                vec3 lightColor3 = vec3(0.5f,0.5f,0.5f)*sunlightSTR;


                float constant = 1.0;
                float linear = 0.09;
                float quadratic = 0.032;


                out vec4 outColor;
                void main()
                {

                    // compute tangent T and bitangent B
                    vec3 Q1 = dFdx(FragPos);
                    vec3 Q2 = dFdy(FragPos);
                    vec2 st1 = dFdx(newTexture);
                    vec2 st2 = dFdy(newTexture);

                    vec3 T = normalize(Q1*st2.t - Q2*st1.t);
                    vec3 B = normalize(-Q1*st2.s + Q2*st1.s);
                    vec3 N = normalize(Norm);

                    // the transpose of texture-to-eye space matrix
                    mat3 TBN = mat3(T, B, N);

                    ///NORMALS TO WORLDSPACE
                    vec3 Normal = normalize(texture(normalMap, newTexture).xyz * 2.0 - 1.0);
                    Normal = normalize(TBN * Normal);

                    ///DIFFUSE
                    vec3 norm = normalize(vec3(Normal));
                    vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
                    float diff = max(dot(norm, lightDir), 0.0);
                    vec3 diffuse = diff * lightColor;

                    vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
                    float diff2 = max(dot(norm, lightDir2), 0.0);
                    vec3 diffuse2 = diff2 * lightColor2;

                    vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
                    float diff3 = max(dot(norm, lightDir3), 0.0);
                    vec3 diffuse3 = diff3 * lightColor3;


                    vec3 norm9 = normalize(vec3(-Normal));

                    float diff9 = max(dot(norm9, lightDir), 0.0);
                    vec3 diffuse9 = diff9 * lightColor;


                    float diff29 = max(dot(norm9, lightDir2), 0.0);
                    vec3 diffuse29 = diff29 * lightColor2;


                    float diff39 = max(dot(norm9, lightDir3), 0.0);
                    vec3 diffuse39 = diff39 * lightColor3;





                    ////SPECULAR
                    vec3 viewDir = normalize((vec3(ViewPos))-FragPos);
                    vec3 reflectDir = reflect(-lightDir, norm);
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
                    vec3 specular = specularStrenght * spec * lightColor;

                    vec3 reflectDir2 = reflect(-lightDir2, norm);
                    float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
                    vec3 specular2 = specularStrenght * spec2 * lightColor2;

                    vec3 reflectDir3 = reflect(-lightDir3, norm);
                    float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), 32);
                    vec3 specular3 = specularStrenght * spec3 * lightColor3;

                    //ATTENUATION
                    float distance    = length(Lightpos.xyz - FragPos);
                    float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

                    float distance2    = length(Lightpos2.xyz - FragPos);
                    float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

                    float distance3    = length(Lightpos3.xyz - FragPos);
                    float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

                    ///VALOSUMMA
                    vec3 result = (diffuse  + specular + diffuse9) * attenuation;
                    vec3 result2 = (diffuse2 + specular2 + diffuse29) * attenuation2;
                    vec3 result3 = (diffuse3 + specular3 + diffuse39);

                    vec3 totalResult = result + result2 + result3 + ambient;

                    ///TEXTUURI
                    vec4 texel = texture2D(samplerTexture, newTexture);
                    if(texel.a < 0.5)
                    {
                        discard;
                    }

                    ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
                    outColor = texel * vec4(totalResult, 1.0f);

                }
                """

    shader_instance = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return shader_instance
def instanssi_shaderi_wind_ref():
    vertex_shader = """
                #version 460
                in layout ( location = 0 ) vec3 position;
                in layout ( location = 1 ) vec2 tex;
                in layout ( location = 2 ) vec3 normal;
                in layout ( location = 4 ) mat4 instanceMatrix;
                in layout ( location = 8 ) mat4 instanceTrans;
                in layout ( location = 12 ) float random;


                uniform mat4 projection;
                uniform mat4 view;
                uniform float time;
                uniform float Voima;


                const vec4 plane = vec4(0,1,0,0);


                out vec2 newTexture;
                out vec3 FragPos;
                out vec3 Norm;

                void main()
                {
                    vec3 offset = vec3( Voima * (cos( time/1.5+random *2 ) * (tex.y)) * sin( position.x),
                           0.25 * sin( 1.5*position.x + 2.0*position.z )* (tex.y),
                          Voima * (cos( time/2.0 +random*2 ) * (tex.y)) * sin( position.z)
                         );

                    vec4 v = vec4( position.x + offset.x, position.y + offset.y, position.z + offset.z + offset.x/5.0 + offset.y/2.0, 1.0);// vec4(position,1.0f);
                    gl_Position = projection * view * instanceMatrix * instanceTrans * v;
                    FragPos = vec3(instanceMatrix *instanceTrans *  v);

                    gl_ClipDistance[0] = dot((vec4(FragPos,1.0f)),plane);


                    Norm = normal;
                    newTexture = tex;
                }
                """

    fragment_shader = """
                #version 460
                in vec2 newTexture;
                in vec3 FragPos;
                in vec3 Norm;

                uniform mat4 lightpos;
                uniform mat4 viewPos;
                uniform sampler2D samplerTexture;
                uniform float luhtu;


                vec4 Lightpos = lightpos * vec4(1.0, 1.0, 1.0, 1.0);
                vec4 ViewPos = viewPos * vec4(1.0, 1.0, 1.0, 1.0);
                vec3 lightColor = vec3(0.9f,0.55f,0.2f)*luhtu;
                uniform float ambientSTR;
                vec3 ambient = vec3(0.2f,0.2f,0.15f)*ambientSTR;


                vec4 Lightpos2 = vec4(49.0, 4.0, 9.0, 1.0);
                vec3 lightColor2 = vec3(1.0f,0.0f,0.0f);

                vec4 Lightpos3 = vec4(500.0, 500.0, 500.0, 1.0);
                uniform float sunlightSTR;
                vec3 lightColor3 = vec3(0.5f,0.5f,0.5f)*sunlightSTR;


                float constant = 1.0;
                float linear = 0.09;
                float quadratic = 0.032;


                out vec4 outColor;
                void main()
                {

                    ///DIFFUSE
                    vec3 norm = normalize(vec3(Norm));
                    vec3 lightDir = normalize((vec3(Lightpos)) - FragPos);
                    float diff = max(dot(norm, lightDir), 0.0);
                    vec3 diffuse = diff * lightColor;

                    vec3 lightDir2 = normalize((vec3(Lightpos2)) - FragPos);
                    float diff2 = max(dot(norm, lightDir2), 0.0);
                    vec3 diffuse2 = diff2 * lightColor2;

                    vec3 lightDir3 = normalize((vec3(Lightpos3)) - FragPos);
                    float diff3 = max(dot(norm, lightDir3), 0.0);
                    vec3 diffuse3 = diff3 * lightColor3;


                    vec3 norm9 = normalize(vec3(-Norm));
                    float diff9 = max(dot(norm9, lightDir), 0.0);
                    vec3 diffuse9 = diff9 * lightColor;
                    float diff29 = max(dot(norm9, lightDir2), 0.0);
                    vec3 diffuse29 = diff29 * lightColor2;
                    float diff39 = max(dot(norm9, lightDir3), 0.0);
                    vec3 diffuse39 = diff39 * lightColor3;


                    //ATTENUATION
                    float distance    = length(Lightpos.xyz - FragPos);
                    float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));

                    float distance2    = length(Lightpos2.xyz - FragPos);
                    float attenuation2 = 1.0 / (constant + linear * distance2 + quadratic * (distance2 * distance2));

                    float distance3    = length(Lightpos3.xyz - FragPos);
                    float attenuation3 = 1.0 / (constant + linear * distance3 + quadratic * (distance3 * distance3));

                    ///VALOSUMMA
                    vec3 result = (diffuse  + diffuse9) * attenuation;
                    vec3 result2 = (diffuse2 + diffuse29) * attenuation2;
                    vec3 result3 = (diffuse3 + diffuse39);

                    vec3 totalResult = result + result2 + result3 + ambient;

                    ///TEXTUURI
                    vec4 texel = texture2D(samplerTexture, newTexture);
                    if(texel.a < 0.5)
                    {
                        discard;
                    }

                    ///MIXAILLAAN SE FAKING FINAL PIXEL VARI ULOS
                    outColor = texel * vec4(totalResult, 1.0f);

                }
                """

    shader_instance = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader,
                                        GL_FRAGMENT_SHADER))
    return shader_instance


#tkinterGUI :D
def teekoo(asetusarvo):
    root = tk.Tk()
    root.title("GUI :D vittu tk GUI")
    root.iconbitmap("ikoni.ico")



    def get_values():
        try:
            asetusarvo.Max_nopeus = float(entry_A.get())
        except:
            pass
        try:
            asetusarvo.jump = float(entry_B.get())
        except:
            pass
        try:
            asetusarvo.sens = float(entry_C.get())
        except:
            pass
        try:
            asetusarvo.AL = float(entry_D.get())
        except:
            pass
        try:
            asetusarvo.luhtu_strena = float(entry_E.get())
        except:
            pass
        try:
            asetusarvo.SL = float(entry_F.get())
        except:
            pass




        if int(invy.get()) == 1:
            asetusarvo.invert_y = False
        else:
            asetusarvo.invert_y = True

        if int(posp.get()) == 1:
            asetusarvo.PP = 1
        else:
            asetusarvo.PP= 0

        if int(mob.get()) == 1:
            asetusarvo.MB = 1
        else:
            asetusarvo.MB= 0

        if int(mult.get()) == 1:
            asetusarvo.terrain_Mtex_Mnorm = 1
        else:
            asetusarvo.terrain_Mtex_Mnorm= 0

        if int(glow.get()) == 1:
            asetusarvo.GLOW = 1
        else:
            asetusarvo.GLOW= 0

        asetusarvo.Evolume = Evolume.get()/100
        asetusarvo.Mvolume = Mvolume.get()/100

    def reset():
        entry_A.delete(0, tk.END)
        entry_A.insert(tk.END, 7)
        asetusarvo.Max_nopeus = 7

        entry_B.delete(0, tk.END)
        entry_B.insert(tk.END, 0.1)
        asetusarvo.jump = 0.1

        entry_C.delete(0, tk.END)
        entry_C.insert(tk.END, 0.08)
        asetusarvo.sens = 0.08

        entry_D.delete(0, tk.END)
        entry_D.insert(tk.END, 0.0)
        asetusarvo.AL = 0.0

        entry_E.delete(0, tk.END)
        entry_E.insert(tk.END, 1.5)
        asetusarvo.luhtu_strena = 1.5

        entry_E.delete(0, tk.END)
        entry_E.insert(tk.END, 0.0)
        asetusarvo.SL = 0.0


    fontti = "Verdana 16 bold"

    teksti_1X = tk.Label(root, text="         TESTIKS VAA", font=fontti,fg="red")

    teksti_11 = tk.Label(root, text="Max_nopeus                ",font=fontti)
    teksti_12 = tk.Label(root, text="Hypyn voimakkuus          ",font=fontti)
    teksti_13 = tk.Label(root, text="sens                      ",font=fontti)
    teksti_14 = tk.Label(root, text="Ambient valo              ",font=fontti)
    teksti_15 = tk.Label(root, text="Oman valon voimakkuus     ",font=fontti)
    teksti_16 = tk.Label(root, text="Auringon valo             ",font=fontti)

    teksti_17 = tk.Label(root, text="musavola             ", font=fontti)
    teksti_18 = tk.Label(root, text="muu vola             ", font=fontti)

    teksti_2X = tk.Label(root, text="        OHJEET",font=fontti,fg="red")
    teksti_2Y = tk.Label(root, text="              ",font=fontti)

    teksti_21 = tk.Label(root, text="NÄPPÄIN        ",font=fontti)
    teksti_22 = tk.Label(root, text="WASD ja mouse  ",font=fontti)
    teksti_23 = tk.Label(root, text="F              ",font=fontti)
    teksti_24 = tk.Label(root, text="P              ",font=fontti)
    teksti_25 = tk.Label(root, text="G              ",font=fontti)
    teksti_26 = tk.Label(root, text="ESC            ",font=fontti)
    teksti_27 = tk.Label(root, text="SPACE          ", font=fontti)
    teksti_28 = tk.Label(root, text="               ", font=fontti)
    teksti_29 = tk.Label(root, text="Ruksilla paasee tasta ikkunasta", font=fontti,fg="red")

    teksti_31 = tk.Label(root, text="TOIMINTO      ",font=fontti)
    teksti_32 = tk.Label(root, text="liikkuminen   ",font=fontti)
    teksti_33 = tk.Label(root, text="valo          ",font=fontti)
    teksti_34 = tk.Label(root, text="pause         ",font=fontti)
    teksti_35 = tk.Label(root, text="TÄÄ ikkuna      ",font=fontti)
    teksti_36 = tk.Label(root, text="quit          ",font=fontti)
    teksti_37 = tk.Label(root, text="hyppy          ", font=fontti)


    entry_A = tk.Entry(root,font=fontti)
    entry_A.insert(tk.END, asetusarvo.Max_nopeus)
    entry_B = tk.Entry(root,font=fontti)
    entry_B.insert(tk.END, asetusarvo.jump)
    entry_C = tk.Entry(root,font=fontti)
    entry_C.insert(tk.END, asetusarvo.sens)
    entry_D = tk.Entry(root,font=fontti)
    entry_D.insert(tk.END, asetusarvo.AL)
    entry_E = tk.Entry(root,font=fontti)
    entry_E.insert(tk.END, asetusarvo.luhtu_strena)
    entry_F = tk.Entry(root, font=fontti)
    entry_F.insert(tk.END, asetusarvo.SL)

    teksti_1X.grid(row=0, column=0, sticky=tk.E)

    teksti_11.grid(row=1, column=1, sticky=tk.W)
    teksti_12.grid(row=2, column=1, sticky=tk.W)
    teksti_13.grid(row=3, column=1, sticky=tk.W)
    teksti_14.grid(row=4, column=1, sticky=tk.W)
    teksti_15.grid(row=5, column=1, sticky=tk.W)
    teksti_16.grid(row=6, column=1, sticky=tk.W)

    teksti_17.grid(row=10, column=3, sticky=tk.W)
    teksti_18.grid(row=11, column=3, sticky=tk.W)

    teksti_2X.grid(row=0, column=2, sticky=tk.E)
    teksti_2Y.grid(row=0, column=3, sticky=tk.E)

    teksti_21.grid(row=1, column=2, sticky=tk.W)
    teksti_22.grid(row=2, column=2, sticky=tk.W)
    teksti_23.grid(row=3, column=2, sticky=tk.W)
    teksti_24.grid(row=4, column=2, sticky=tk.W)
    teksti_25.grid(row=5, column=2, sticky=tk.W)
    teksti_26.grid(row=6, column=2, sticky=tk.W)
    teksti_27.grid(row=7, column=2, sticky=tk.W)
    teksti_28.grid(row=8, column=2, sticky=tk.W)
    teksti_29.grid(row=9, column=2, sticky=tk.W)

    teksti_31.grid(row=1, column=3, sticky=tk.W)
    teksti_32.grid(row=2, column=3, sticky=tk.W)
    teksti_33.grid(row=3, column=3, sticky=tk.W)
    teksti_34.grid(row=4, column=3, sticky=tk.W)
    teksti_35.grid(row=5, column=3, sticky=tk.W)
    teksti_36.grid(row=6, column=3, sticky=tk.W)
    teksti_37.grid(row=7, column=3, sticky=tk.W)

    entry_A.grid(row=1, column=0)
    entry_B.grid(row=2, column=0)
    entry_C.grid(row=3, column=0)
    entry_D.grid(row=4, column=0)
    entry_E.grid(row=5, column=0)
    entry_F.grid(row=6, column=0)



    button_1 = tk.Button(root, text="set",font=fontti, command=get_values,fg="red")
    button_1.grid(row=20, column=0)
    button_2 = tk.Button(root, text="reset",font=fontti, command=reset,fg="red")
    button_2.grid(row=20, column=1, sticky=tk.W)

    #hieno on
    if asetusarvo.invert_y:
        asd = 0
    else:
        asd = 1

    invy = tk.IntVar(value=asd)
    checkbutton1 = tk.Checkbutton(root,text="INVERT MOUSE",font=fontti,variable=invy)
    checkbutton1.grid(row=7, column=0)

    posp = tk.IntVar(value=asetusarvo.PP)
    checkbutton1 = tk.Checkbutton(root, text="postprocesstest", font=fontti, variable=posp)
    checkbutton1.grid(row=8, column=0)

    mob = tk.IntVar(value=asetusarvo.MB)
    checkbutton1 = tk.Checkbutton(root, text="paska_motionblur", font=fontti, variable=mob)
    checkbutton1.grid(row=9, column=0)

    mult = tk.IntVar(value=asetusarvo.terrain_Mtex_Mnorm)
    checkbutton1 = tk.Checkbutton(root, text="terrain multitext & normal", font=fontti, variable=mult)
    checkbutton1.grid(row=10, column=0)

    glow = tk.IntVar(value=asetusarvo.GLOW)
    checkbutton1 = tk.Checkbutton(root, text="glow", font=fontti, variable=glow)
    checkbutton1.grid(row=11, column=0)

    Evolume = tk.Scale(root, from_=0, to=100,orient=tk.HORIZONTAL)
    Evolume.grid(row=11, column=2,sticky=tk.E)
    Evolume.set(asetusarvo.Evolume*100)

    Mvolume = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
    Mvolume.grid(row=10, column=2,sticky=tk.E)
    Mvolume.set(asetusarvo.Mvolume*100)

    root.mainloop()


#network
def nettihomma(client,severi,hk0,hahmoY,hk2,c2x,c2y,c2z):
    Clock = pygame.time.Clock()
    tLock = threading.Lock()
    shutdown = False
    def receving(name, sock):
        while not shutdown:
            try:
                tLock.acquire()
                while True:
                    data, addr = sock.recvfrom(1024)
                    lista = data.decode().split(",")
                    c2x.value = float(lista[0])
                    c2y.value = float(lista[1])-1
                    c2z.value = float(lista[2])
                    #print([c2x.value,c2y.value,c2z.value])
            except:
                pass
            finally:
                tLock.release()
                Clock.tick(20)

    host = client
    port = 0

    server = (severi,2620)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    s.setblocking(0)

    rT = threading.Thread(target=receving, args=("RecvThread",s))
    rT.start()



    while True:
        message = str(str(-hk0.value)+","+str(-hahmoY.value)+","+str(-hk2.value))
        if message != '':
            s.sendto(message.encode(), server)
        Clock.tick(20)
        tLock.acquire()

        tLock.release()

    rT.join()
    s.close()

def podsixFAG(hk0,hahmoY,hk2,c2x,c2y,c2z):
    class GAME():
        def __init__(self):
            while True:
                self.omapos = [hk0.value, hahmoY.value, hk2.value]
                self.client2pos = [c2x.value,c2y.value,c2z.value]
    GAME()

if __name__ == "__main__":
    multiprocessing.freeze_support()

    jono = multiprocessing.Queue()  # odottaa puttia aina
    jono2 = multiprocessing.Queue()

    hk0 = multiprocessing.Value('d', 0.0)
    hk1 = multiprocessing.Value('d', 0.0)
    hk2 = multiprocessing.Value('d', 0.0)
    sulje = multiprocessing.Value('d', 0.0)

    deny_jump = multiprocessing.Value('d', 0.0)
    Ctype = multiprocessing.Value('d', 0.0)
    collision = multiprocessing.Value('d', 0.0)
    hahmoY = multiprocessing.Value('d', 0.0)
    hcy = multiprocessing.Value('d', 0.0)

    c2x = multiprocessing.Value('d', 0.0)
    c2y = multiprocessing.Value('d', 0.0)
    c2z = multiprocessing.Value('d', 0.0)

    client = input("oma IP:->")
    severi = input("server IP:->")

    p1 = multiprocessing.Process(target=main, args=(jono, hk0, hk1, hk2, sulje, jono2, hahmoY, collision, Ctype, deny_jump,hcy,c2x,c2y,c2z,))
    p2 = multiprocessing.Process(target=etitaangrid_multi, args=(jono, hk0, hk1, hk2,sulje,))
    p3 = multiprocessing.Process(target=collision_check_multi, args=(jono2, hk0, hk2, hahmoY, collision, Ctype, deny_jump,sulje,hcy,))
    p4 = multiprocessing.Process(target=nettihomma, args=(client,severi,hk0,hahmoY,hk2,c2x,c2y,c2z,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    #todella kämänen tyyli kattoo onko main elossa :D
    while p1.is_alive():
        pygame.time.wait(1000)
    p2.terminate()
    p3.terminate()
    p4.terminate()
